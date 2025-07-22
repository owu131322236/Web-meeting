import pandas as pd
import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime
import numpy as np
import time
import accelerate
from transformers import AutoModel, AutoTokenizer
import streamlit as st
import speech_recognition as sr
import av
import collections
# cl-tohoku/bert-base-japanese-whole-word-masking
recognition_queue = collections.deque()
model_name = "intfloat/multilingual-e5-small"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def make_llm_output(messages, max_new_tokens=2048):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

class MeetingAssistant:
    def __init__(self, db_path = 'meeting_topics.db'):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        conn = sqlite3.connect(self.db_path) #データベースを開く
        cursor = conn.cursor() # データベースに書く準備
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                meeting_id TEXT PRIMARY KEY, 
                topic_text TEXT NOT NULL,
                keywords TEXT
            )
        ''')
        conn.commit() #書いたことをgitみたいにcommit
        conn.close() #データベースを閉じる

    def add_topic(self, meeting_id, topic_text, keywords=None): #新しいテーマ用のデータベース
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            keywords_str = ','.join(keywords) if keywords else '' #キーワードで入力されたものを一つに繋げる
            cursor.execute("INSERT OR REPLACE INTO topics (meeting_id, topic_text,keywords)VALUES(?, ?, ?)",
                            (meeting_id, topic_text, keywords_str)) #表にテーマを書き込む
            conn.commit()
            conn.close()

    def get_topic(self, meeting_id): #テーマをゲットする関数
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor() #データベースに書く準備
        cursor.execute("SELECT topic_text, keywords FROM topics WHERE meeting_id = ?", (meeting_id,)) #会議の番号からテーマを探す
        result = cursor.fetchone() #見つかったテーマを取り出す
        conn.close() #データベースを閉じる
        if result: #もしテーマが見つかったら
            topic_text, keywords_str = result
            keywords = keywords_str.split(',') if keywords_str else [] #言葉をバラバラに戻す
            return {"topic_text": topic_text, "keywords": keywords}
        return None
            
topic_manager = MeetingAssistant()
topic_manager.add_topic("meeting_001", "新製品Xのマーケティング戦略について議論", ["新製品X", "マーケティング", "戦略"])  

from transformers import AutoModel, AutoTokenizer #以下のclassで使う機能を呼び出す

r = sr.Recognizer()
mic = sr.Microphone()

class TextEmbedder: #受け取ったテキストをパソコンにわかりやすく渡すクラス本体
    def __init__(self, model_name="cl-tohoku/bert-base-japanese-whole-word-masking"): #BARTをモデルとして利用！
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) #受け取った言葉をバラバラにする
        self.model = AutoModel.from_pretrained(model_name) #バラバラになった言葉をコンピューターにわかる形にする
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPUの速度を調べる
        self.model.to(self.device) #GPUで動かす準備
        self.model.eval()

    def get_embedding(self,text):
        inputs = self.tokenizer(text, return_tensors ="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()} #GPUに言葉を送る

        with torch.no_grad():
            outputs = self.model(**inputs) # 中身を渡し、コンピューターに分かりやすい形で書き直す


        #先ほどのもとをまとめる
        attention_mask = inputs['attention_mask']
        expanded_mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * expanded_mask, 1)
        sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
        mean_embedding =sum_embeddings / sum_mask #まとめる
    
        return mean_embedding.squeeze(0).cpu().numpy()

    def calculate_similarity(self, embedding1, embedding2):
        from sklearn.metrics.pairwise import cosine_similarity 
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
    
class TopicDeviationDetector: #ミーティングの監視役
    def __init__(self, topic_embedding: np.ndarray, topic_info: dict, similarity_threshold: float =0.7, consecutive_deviations_needed: int =2, cooldown_period_seconds: int =10): #準備
        self.topic_embedding = topic_embedding 
        self.topic_info = topic_info
        self.similarity_threshold = similarity_threshold #関連度の限度
        self.consecutive_deviations_needed = consecutive_deviations_needed #何回続けてボーダーラインを超えたら警告するか
        self.consecutive_deviations_count = 0 #ボーダーラインを超えた数
        self.last_notification_time = 0 #最後の警告時間
        self.cooldown_period_seconds = cooldown_period_seconds #クールダウンのタイム
        self.text_embedder = TextEmbedder() #コンピュータに渡すデーターへの変換

    def process_utterance(self, current_utterance_text): #ミーティング監視のメイン
        current_time = time.time() #今の時間を知る

        if current_time - self.last_notification_time < self.cooldown_period_seconds:
            return False, "クールダウン中" 

        utterance_embedding = self.text_embedder.get_embedding(current_utterance_text) #今の発言をコンピューター用に変換
        similarity = self.text_embedder.calculate_similarity(self.topic_embedding, utterance_embedding) #関連性を調べる

        print(f"今のお話: '{current_utterance_text}'")
        print(f"テーマとの似ている度合い: {similarity:.4f} (ボーダーライン: {self.similarity_threshold})")

        if similarity < self.similarity_threshold: 
            self.consecutive_deviations_count += 1 #非関連話題であるカウンターを1追加
            if self.consecutive_deviations_count >= self.consecutive_deviations_needed:
                self.last_notification_time = current_time #警告
                self.consecutive_deviations_count = 0 #カウンターをゼロに
                # 「おーい！」って言うメッセージを作る
                return True, f"話が逸れています！ 今のお話は'{current_utterance_text[:30]}...'ですが、テーマは'{self.topic_info['topic_text']}'です！"
            else:
                return False, f"お題がら派生しています。 (あと {self.consecutive_deviations_needed - self.consecutive_deviations_count}回話が逸れた場合は注意します)"
        else:
            self.consecutive_deviations_count = 0 
            return False, "テーマに沿って話していますね！"

class AudioRecoginizer:
    
    @staticmethod
    def audio_frame_callback_func(frame: av.AudioFrame) -> av.AudioFrame:
        print("Received audio frame!")
        if st.session_state.detector is None:
            return frame
        
        # recoginition_queueが未初期化の場合に初期化
        if 'recoginition_queue' not in st.session_state:
            st.session_state.recoginition_queue = collections.deque()
            
        audio_data_np = frame.to_ndarray(format="s16").flatten() 
        audio_data_sr = sr.AudioData(audio_data_np.tobytes(), frame.sample_rate, 2) # 2はsample_width (s16は2バイト)
        st.session_state.recoginition_queue.append(audio_data_sr)

        return frame
    
    @staticmethod
    def recoginizer_thread():
        global r 
        print("recoginizer_threadは動いています", flush=True)
        # while st.session_state.recoginizer_thread_running:
        while True:
            print("recoginizer_thread_runningは動いています", flush=True)
            if  not recognition_queue:
                # キューが初期化されていない場合は待機
                time.sleep(0.1)
                print("ただいま待機中", flush=True)
                continue
        
            audio_data = recognition_queue.popleft()
            print(audio_data, flush=True)
            try:
                audio_text = r.recognize_google(audio_data, language='ja-JP') 
                if audio_text:
                    # 発言履歴に追加
                    print("st.session_state.all_utterances.append(audio_text)")
                    
                    # テーマ逸脱検知
                    if st.session_state.detector:
                        is_deviated, message = st.session_state.detector.process_utterance(audio_text)
                        if is_deviated:
                            st.session_state.all_utterances.append(f"AIアシスタント (逸脱): {message}")
                        else:
                            # 発言がテーマに沿っている場合のメッセージは、必ずしもチャットに表示する必要はないが、
                            # 現在のコードではすべてのAIアシスタントメッセージが追加される
                            st.session_state.all_utterances.append(f"AIアシスタント: {message}")

                    st.experimental_rerun() 
            except sr.UnknownValueError:
                print("音声が認識できませんでした")
                
            except sr.RequestError as e:
                print(f"Google Speech Recognitionサービスへのリクエスト中にエラーが発生しました; {e}") 
            except Exception as e: # その他の予期せぬエラー
                print(f"音声認識スレッドで予期せぬエラーが発生しました: {e}")
        else:
            time.sleep(0.1) # キューが空の場合は少し待機