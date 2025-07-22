import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration
import collections
import time
import speech_recognition as sr
import threading 
import av
import pydub
import numpy as np
import openai

r = sr.Recognizer() # グローバルでRecognizerインスタンスを生成

from backend_app_preparation import MeetingAssistant,TextEmbedder, TopicDeviationDetector, AudioRecoginizer

st.set_page_config(
    page_title="FlowLink",
    page_icon="🗣️"
)

st.title("🗣️FlowLink")
st.write("会議支援ツール ---どんな人もスムーズな会議の進行を")

if "recoginizer_thread_running" not in st.session_state:
    st.session_state.recoginizer_thread_running = False

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()[0]
        self.buffer.append(audio)
        return frame  # フレームは変更しない


if 'topic_info' not in st.session_state:
    st.session_state.topic_info = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'all_utterances' not in st.session_state:
    st.session_state.all_utterances = collections.deque(maxlen=20)
if 'recoginition_queue' not in st.session_state: # recoginition_queueを初期化
    st.session_state.recoginition_queue = collections.deque()
if 'recognizer_thread_running' not in st.session_state:
    st.session_state.recognizer_thread_running = False


tab_titles = ['会議の情報','会議チャット','発言記録']
tab1, tab2, tab3 = st.tabs(tab_titles)

with tab1:
    with st.expander("会議の最終目標", expanded=True):
        topic_thema = st.text_input("会議のテーマ", key="topic_thema")
        topic_text_input = st.text_input("会議の目的", key="topic_input_area")
        set_topic_button = st.button("決定", key="set_topic_button")
        if set_topic_button or st.session_state.topic_info is None:
            if topic_text_input:
                topic_manager = MeetingAssistant()
                meeting_id = "current_meeting"
                topic_manager.add_topic(meeting_id, topic_text_input)
                st.session_state.topic_info = topic_manager.get_topic(meeting_id)

                @st.cache_resource
                #Detector に渡すために TextEmbedder のインスタンスは作らない。ただし、topic_embedding を生成するためには TextEmbedder が必要なのでインスタンスは作成
                def get_detector_instance(current_topic_info_dict):
                    embedder_instance = TextEmbedder()
                    topic_embedding =embedder_instance.get_embedding(current_topic_info_dict['topic_text'])
                    detector_instance = TopicDeviationDetector(
                        topic_embedding,  
                        current_topic_info_dict,       
                        similarity_threshold=0.8,
                        consecutive_deviations_needed=2,
                        cooldown_period_seconds=10
                    )
                    return detector_instance
                
                st.session_state.detector = get_detector_instance(st.session_state.topic_info)

                st.success(f"会議の情報が設定されました: {st.session_state.topic_info['topic_text']}")
            else:
                st.warning("情報を入力してください。")
with tab2:
    st.info("マイクをオンにしてください")

with tab3:
    
    st.write('音声認識を開始します。マイクに話しかけてください。') # 'test'をより分かりやすいメッセージに
    webrtc_ctx = webrtc_streamer(
        key="audio-demo2",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=AudioRecoginizer.audio_frame_callback_func, 
        media_stream_constraints={"video": False, "audio": True},
    )
    result_placeholder = st.empty()
   
    # if  webrtc_ctx.state.playing:
    #     if not st.session_state.recognizer_thread_running: 
    #         st.session_state.recognizer_thread_running = True
    #         # # threading.Threadのtargetにはrを渡す必要がない
    #         threading.Thread(target=AudioRecoginizer.recoginizer_thread, daemon=True).start()
    #         # audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    #     result_placeholder.info("マイクがオンです。発言を認識中")
    # else:
    #     if st.session_state.recognizer_thread_running:
    #         st.session_state.recognizer_thread_running = False
    #     result_placeholder.write("マイクが音声を認識していません")
    
    st.subheader("最近の発言履歴")
    # 発言履歴をリアルタイムで表示
    if st.session_state.all_utterances:
        for i, utt in enumerate(reversed(list(st.session_state.all_utterances))):
            st.markdown(f"**{len(st.session_state.all_utterances) - i}.** {utt}")
    else:
        st.write("まだ発言はありません。")
