# -*- coding: utf-8 -*-

import os
import json
import subprocess
from datetime import timedelta
from faster_whisper import WhisperModel
from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
import shutil
warnings.filterwarnings('ignore')

# 형태소 분석
try:
    from kiwipiepy import Kiwi
    USE_KIWI = True
except ImportError:
    try:
        from konlpy.tag import Okt
        USE_KIWI = False
    except ImportError:
        USE_KIWI = None


class ShortsAnalyzer:
    """쇼츠 영상 분석 및 필터링 클래스"""
    
    def __init__(self, community_keywords=None):
        self.community_keywords = community_keywords or []
        
        if USE_KIWI:
            print("🔧 Kiwi 형태소 분석기 로딩...")
            self.kiwi = Kiwi()
        elif USE_KIWI is False:
            print("🔧 Okt 형태소 분석기 로딩...")
            self.okt = Okt()
        
        self.positive_words = set([
            '좋다', '최고', '대박', '재밌다', '멋지다', '훌륭하다', '완벽', '감동',
            '행복', '사랑', '웃다', '기쁘다', '즐겁다', '유쾌', '흥미롭다',
            '놀랍다', '신기하다', '환상적', '굉장하다', '탁월하다', '예쁘다',
            '아름답다', '귀엽다', '달콤하다', '맛있다', '신나다', '화려하다'
        ])
        
        self.negative_words = set([
            '나쁘다', '최악', '짜증', '지루하다', '실망', '별로', '후회',
            '슬프다', '우울', '화나다', '미치다', '힘들다', '아프다', '불편',
            '싫다', '무섭다', '걱정', '문제', '실패', '끔찍하다', '더럽다'
        ])
        
        print(f"✅ 분석기 초기화 완료 (커뮤니티 키워드: {len(self.community_keywords)}개)")
    
    def extract_morphemes(self, text):
        """형태소 분석 및 명사/동사 추출"""
        if USE_KIWI:
            result = self.kiwi.analyze(text)
            morphemes = []
            for token in result[0][0]:
                if token.tag in ['NNG', 'NNP', 'VV', 'VA']:
                    morphemes.append(token.form)
            return morphemes
        elif USE_KIWI is False:
            return self.okt.nouns(text) + [word for word, pos in self.okt.pos(text) if pos in ['Verb', 'Adjective']]
        else:
            return re.findall(r'[\w가-힣]+', text)
    
    def clean_words(self, words):
        """불용어 제거 및 단어 정제"""
        stop_words = set([
            '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', 
            '도', '으로', '로', '에서', '하다', '있다', '되다', '이다',
            '그', '저', '것', '수', '등', '및', '좀', '막', '진짜', '되게',
            '거', '게', '네', '요', '음', '어', '아', '야', '임', '들',
            '때', '정도', '만', '부터', '까지', '마다', '조차', '나', '이런',
            '같다', '보다', '위하다', '대하다', '통하다', '관하다'
        ])
        
        cleaned = []
        for word in words:
            if len(word) >= 2 and word not in stop_words:
                cleaned.append(word)
        
        return cleaned
    
    def analyze_words(self, text):
        """텍스트에서 단어 추출 및 정제"""
        morphemes = self.extract_morphemes(text)
        cleaned_words = self.clean_words(morphemes)
        word_freq = Counter(cleaned_words)
        
        return {
            'words': cleaned_words,
            'word_frequency': dict(word_freq.most_common(20)),
            'unique_words': len(word_freq),
            'total_words': len(cleaned_words)
        }
    
    def calculate_community_match(self, short_words):
        """커뮤니티 키워드와의 매칭도 계산"""
        if not self.community_keywords:
            return 0.0, []
        
        community_set = set([kw.lower() for kw in self.community_keywords])
        short_set = set([w.lower() for w in short_words])
        
        matched_keywords = community_set.intersection(short_set)
        
        if len(community_set) == 0:
            match_score = 0.0
        else:
            match_score = len(matched_keywords) / len(community_set)
        
        return match_score, list(matched_keywords)
    
    def sentiment_analysis(self, text, words):
        """감정 분석 (긍정/부정/중립)"""
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 'neutral', 0.0
        
        positive_ratio = positive_count / total_sentiment_words
        
        if positive_ratio > 0.6:
            sentiment = 'positive'
        elif positive_ratio < 0.4:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        sentiment_score = (positive_count - negative_count) / max(len(words), 1)
        
        return sentiment, sentiment_score
    
    def topic_modeling(self, texts, n_topics=5):
        """주제 모델링 (LDA)"""
        if len(texts) < n_topics:
            n_topics = max(1, len(texts))
        
        vectorizer = TfidfVectorizer(
            max_features=100,
            min_df=1,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20
            )
            lda.fit(tfidf_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-5:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'keywords': top_words
                })
            
            doc_topics = lda.transform(tfidf_matrix)
            doc_topic_assignments = doc_topics.argmax(axis=1)
            
            return topics, doc_topic_assignments
        
        except Exception as e:
            print(f"⚠️  주제 모델링 실패: {e}")
            return [], [0] * len(texts)
    
    def generate_title(self, text, keywords, sentiment, topic_words):
        """제목 자동 생성"""
        main_keywords = keywords[:3] if keywords else []
        
        sentiment_prefix = {
            'positive': '💡',
            'negative': '⚠️',
            'neutral': '📌'
        }
        prefix = sentiment_prefix.get(sentiment, '📌')
        
        if topic_words:
            title_base = ' '.join(topic_words[:2])
        elif main_keywords:
            title_base = ' '.join(main_keywords)
        else:
            words = text.split()[:10]
            title_base = ' '.join(words)
        
        if len(title_base) > 30:
            title_base = title_base[:30] + '...'
        
        title = f"{prefix} {title_base}"
        
        return title


class ShortsBatchProcessor:
    """쇼츠 폴더 배치 처리 클래스"""
    
    def __init__(self, shorts_folder, output_dir="filtered_shorts", 
                 community_keywords=None, match_threshold=0.2):
        """
        Args:
            shorts_folder: 쇼츠 영상들이 있는 폴더 경로
            output_dir: 필터링된 쇼츠를 저장할 폴더
            community_keywords: 커뮤니티 키워드 리스트
            match_threshold: 필터링 임계값
        """
        self.shorts_folder = shorts_folder
        self.output_dir = output_dir
        self.match_threshold = match_threshold
        os.makedirs(output_dir, exist_ok=True)
        
        # 분석기 초기화
        self.analyzer = ShortsAnalyzer(community_keywords=community_keywords)
        
        # Whisper 모델 로드
        print("🤖 Whisper 음성 인식 모델 로드 중...")
        self.whisper_model = WhisperModel(
            "base", 
            device="cpu", 
            compute_type="int8"
        )
        
        print(f"✅ 배치 처리기 초기화 완료")
        print(f"   입력 폴더: {shorts_folder}")
        print(f"   출력 폴더: {output_dir}")
    
    def get_video_files(self):
        """폴더에서 비디오 파일 목록 가져오기"""
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm']
        video_files = []
        
        for filename in os.listdir(self.shorts_folder):
            file_path = os.path.join(self.shorts_folder, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in video_extensions:
                    video_files.append(file_path)
        
        return sorted(video_files)
    
    def get_video_duration(self, video_path):
        """비디오 길이 가져오기"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            return float(info['format']['duration'])
        except:
            return 0.0
    
    def transcribe_video(self, video_path):
        """비디오 음성 인식"""
        try:
            segments_generator, info = self.whisper_model.transcribe(
                video_path,
                language="ko",
                word_timestamps=True,
                vad_filter=True,
                beam_size=5
            )
            
            segments = []
            for segment in segments_generator:
                segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                })
            
            full_text = ' '.join([seg['text'] for seg in segments])
            
            return {
                'segments': segments,
                'text': full_text,
                'language': info.language
            }
        
        except Exception as e:
            print(f"      ⚠️  음성 인식 실패: {e}")
            return None
    
    def analyze_short(self, video_path):
        """단일 쇼츠 분석"""
        filename = os.path.basename(video_path)
        
        print(f"\n   📹 {filename}")
        
        # 1. 음성 인식
        print(f"      🎤 음성 인식 중...")
        transcription = self.transcribe_video(video_path)
        
        if not transcription or not transcription['text'].strip():
            print(f"      ❌ 텍스트를 추출할 수 없습니다")
            return None
        
        text = transcription['text']
        print(f"      📝 텍스트 추출 완료 ({len(text)}자)")
        
        # 2. 단어 분석
        word_analysis = self.analyzer.analyze_words(text)
        words = word_analysis['words']
        
        # 3. 커뮤니티 매칭
        match_score, matched_kws = self.analyzer.calculate_community_match(words)
        
        # 4. 감정 분석
        sentiment, sentiment_score = self.analyzer.sentiment_analysis(text, words)
        
        # 5. 비디오 정보
        duration = self.get_video_duration(video_path)
        
        result = {
            'filename': filename,
            'path': video_path,
            'duration': duration,
            'text': text,
            'word_analysis': word_analysis,
            'match_score': match_score,
            'matched_keywords': matched_kws,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'transcription': transcription
        }
        
        # 결과 출력
        print(f"      🎯 매칭: {match_score:.1%} | 😊 감정: {sentiment} ({sentiment_score:.2f})")
        if matched_kws:
            print(f"      ✅ 키워드: {', '.join(matched_kws[:5])}")
        
        return result
    
    def process_all_shorts(self):
        """모든 쇼츠 처리"""
        video_files = self.get_video_files()
        
        if not video_files:
            print(f"❌ {self.shorts_folder}에서 비디오 파일을 찾을 수 없습니다")
            return [], []
        
        print(f"\n📊 총 {len(video_files)}개의 쇼츠 발견")
        print("="*80)
        
        all_results = []
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] 분석 중...")
            
            result = self.analyze_short(video_path)
            
            if result:
                all_results.append(result)
        
        print("\n" + "="*80)
        print(f"✅ 분석 완료: {len(all_results)}개 성공")
        
        return all_results
    
    def add_topic_modeling(self, results):
        """주제 모델링 수행"""
        if len(results) < 2:
            for result in results:
                result['topic_model_id'] = 0
                result['topic_keywords'] = []
            return results
        
        print(f"\n🧠 주제 모델링 수행 중...")
        
        texts = [r['text'] for r in results]
        n_topics = min(5, max(2, len(results) // 3))
        topics, doc_topics = self.analyzer.topic_modeling(texts, n_topics=n_topics)
        
        for i, result in enumerate(results):
            topic_id = doc_topics[i] if i < len(doc_topics) else 0
            topic_info = topics[topic_id] if topic_id < len(topics) else {'keywords': []}
            
            result['topic_model_id'] = topic_id
            result['topic_keywords'] = topic_info['keywords']
        
        print(f"✅ {len(topics)}개 주제 발견")
        for topic in topics:
            print(f"   주제 {topic['topic_id']+1}: {', '.join(topic['keywords'])}")
        
        return results
    
    def generate_titles(self, results):
        """제목 생성"""
        print(f"\n✍️  제목 생성 중...")
        
        for result in results:
            title = self.analyzer.generate_title(
                text=result['text'],
                keywords=list(result['word_analysis']['word_frequency'].keys()),
                sentiment=result['sentiment'],
                topic_words=result.get('topic_keywords', [])
            )
            result['title'] = title
        
        return results
    
    def filter_and_copy_shorts(self, results):
        """필터링 및 파일 복사"""
        print(f"\n📂 필터링 및 파일 복사 중...")
        print(f"   매칭 임계값: {self.match_threshold:.0%}")
        
        filtered_results = []
        
        for i, result in enumerate(results, 1):
            if result['match_score'] >= self.match_threshold:
                # 새 파일명 생성 (매칭 점수 포함)
                original_name = os.path.splitext(result['filename'])[0]
                original_ext = os.path.splitext(result['filename'])[1]
                new_filename = f"{i:03d}_{result['match_score']:.0%}_{original_name}{original_ext}"
                new_path = os.path.join(self.output_dir, new_filename)
                
                # 파일 복사
                try:
                    shutil.copy2(result['path'], new_path)
                    result['filtered_filename'] = new_filename
                    result['filtered_path'] = new_path
                    filtered_results.append(result)
                    print(f"   ✅ [{i}] {new_filename}")
                except Exception as e:
                    print(f"   ❌ [{i}] 복사 실패: {e}")
            else:
                print(f"   ⏭️  [{i}] {result['filename']} (매칭 {result['match_score']:.1%} < {self.match_threshold:.0%})")
        
        return filtered_results
    
    def _format_time(self, seconds):
        """초를 HH:MM:SS 형식으로 변환"""
        return str(timedelta(seconds=int(seconds)))
    
    def save_reports(self, all_results, filtered_results):
        """리포트 저장"""
        
        # 1. JSON 리포트
        json_path = os.path.join(self.output_dir, "analysis_report.json")
        
        report = {
            'shorts_folder': self.shorts_folder,
            'settings': {
                'match_threshold': self.match_threshold,
                'community_keywords': self.analyzer.community_keywords
            },
            'statistics': {
                'total_shorts': len(all_results),
                'filtered_shorts': len(filtered_results),
                'filter_rate': len(filtered_results) / max(len(all_results), 1)
            },
            'filtered_shorts': [
                {
                    'original_filename': r['filename'],
                    'filtered_filename': r.get('filtered_filename'),
                    'title': r.get('title'),
                    'duration': r['duration'],
                    'match_score': r['match_score'],
                    'matched_keywords': r['matched_keywords'],
                    'sentiment': r['sentiment'],
                    'sentiment_score': r['sentiment_score'],
                    'topic_keywords': r.get('topic_keywords', []),
                    'word_frequency': r['word_analysis']['word_frequency'],
                    'text': r['text']
                }
                for r in filtered_results
            ],
            'all_shorts_summary': [
                {
                    'filename': r['filename'],
                    'match_score': r['match_score'],
                    'sentiment': r['sentiment'],
                    'filtered': r in filtered_results
                }
                for r in all_results
            ]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 JSON 리포트 저장: {json_path}")
        
        # 2. 텍스트 리포트
        txt_path = os.path.join(self.output_dir, "report.txt")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("쇼츠 배치 분석 리포트\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"입력 폴더: {self.shorts_folder}\n")
            f.write(f"전체 쇼츠: {len(all_results)}개\n")
            f.write(f"필터링 통과: {len(filtered_results)}개\n")
            f.write(f"필터링 비율: {len(filtered_results)/max(len(all_results), 1):.1%}\n")
            f.write(f"커뮤니티 키워드: {', '.join(self.analyzer.community_keywords[:20])}\n")
            f.write(f"매칭 임계값: {self.match_threshold:.0%}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("필터링된 쇼츠 상세 정보\n")
            f.write("=" * 80 + "\n")
            
            for i, result in enumerate(filtered_results, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"[{i}] {result.get('title', 'N/A')}\n")
                f.write(f"{'='*80}\n")
                f.write(f"원본 파일: {result['filename']}\n")
                f.write(f"새 파일명: {result.get('filtered_filename')}\n")
                f.write(f"길이: {self._format_time(result['duration'])} ({result['duration']:.1f}초)\n")
                f.write(f"\n📊 분석 결과:\n")
                f.write(f"  - 커뮤니티 매칭: {result['match_score']:.1%}\n")
                f.write(f"  - 매칭된 키워드: {', '.join(result['matched_keywords']) if result['matched_keywords'] else '없음'}\n")
                f.write(f"  - 감정: {result['sentiment']} (점수: {result['sentiment_score']:.2f})\n")
                f.write(f"  - 주제 키워드: {', '.join(result.get('topic_keywords', []))}\n")
                f.write(f"\n🔤 주요 단어:\n")
                for word, freq in list(result['word_analysis']['word_frequency'].items())[:10]:
                    f.write(f"  - {word}: {freq}회\n")
                f.write(f"\n📝 전체 텍스트:\n")
                f.write(f"  {result['text']}\n")
            
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("전체 쇼츠 요약 (매칭 점수순)\n")
            f.write("=" * 80 + "\n\n")
            
            sorted_results = sorted(all_results, key=lambda x: x['match_score'], reverse=True)
            
            f.write(f"{'순위':<6} {'파일명':<40} {'매칭':<8} {'감정':<10} {'필터링'}\n")
            f.write("-" * 80 + "\n")
            
            for i, result in enumerate(sorted_results, 1):
                filtered_mark = "✅" if result in filtered_results else "❌"
                f.write(f"{i:<6} {result['filename']:<40} {result['match_score']:>6.1%}  "
                       f"{result['sentiment']:<10} {filtered_mark}\n")
        
        print(f"📄 텍스트 리포트 저장: {txt_path}")
        
        # 3. CSV 리포트 (간단 버전)
        csv_path = os.path.join(self.output_dir, "summary.csv")
        
        with open(csv_path, 'w', encoding='utf-8-sig') as f:  # Excel 호환을 위해 utf-8-sig
            f.write("순위,파일명,매칭점수,매칭키워드,감정,감정점수,필터링여부,주요단어\n")
            
            sorted_results = sorted(all_results, key=lambda x: x['match_score'], reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                filtered = "통과" if result in filtered_results else "제외"
                matched_kws = '; '.join(result['matched_keywords'])
                top_words = '; '.join(list(result['word_analysis']['word_frequency'].keys())[:5])
                
                f.write(f"{i},{result['filename']},{result['match_score']:.2%},"
                       f"\"{matched_kws}\",{result['sentiment']},{result['sentiment_score']:.2f},"
                       f"{filtered},\"{top_words}\"\n")
        
        print(f"📊 CSV 리포트 저장: {csv_path}")


def main():
    """메인 실행 함수"""
    
    # Segmentation fault 방지
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    try:
        torch.set_num_threads(1)
    except ImportError:
        pass
    
    # ==================== 설정 ====================
    SHORTS_FOLDER = "fast_shorts"  # 쇼츠 영상들이 있는 폴더
    OUTPUT_DIR = "filtered_shorts"  # 필터링된 쇼츠를 저장할 폴더
    
    # 커뮤니티 키워드 설정
    COMMUNITY_KEYWORDS = [
        '커넥순','오늘','오늘 왜케','인형뽑기','기네','바지','바지 질문','바지 질문 저런','바지 질문 저런 대답','처음','반지','반지 가지','금순자아','요랑','요랑 어제','딜교','딜교 잘','같긔','촉촉훈훈강아디','짱개들','왜 또 지랄','오늘 왤케','오늘 왤케 현실','짱깨','끄롬할','끄롬할 수','하필','하필 머학교','하필 머학교 시험기간','오늘 좀','오늘 좀 피곤','진짜','진짜 의젓젤리','인형뽑기 최종','인형','인형 뽑기','인형 뽑기할때','인형 뽑기할때 언니','이런 가오','이런 가오 부릴때','인형뽑을때','요즘','요즘 손','요즘 손 핏줄','글케','글케 좋드','이 눈빛','공격','오늘 막판','턱 괴고','턱 괴고 댓글','턱 괴고 댓글 쳐다볼때','그 얘기','우리보고','우리보고 댓글','한거','언니','언니 머리속','한번','오늘 손','이자벨','이자벨 금순','이자벨 금순과 커넥순','역시','역시 미녀','제철','젤리','대해','복습','메론빵','코엽던뎈','스테이크','띠니','능력','있으면 젤리','있으면 젤리 아닌척','있으면 젤리 아닌척 지원','소찌키','언니 봊돌','대천사','대천사 금순','초딩들','두개','여러모로','여러모로 소신','중년저씨들','능숙한 여자','진짜 어른','어제','어제 못','자기','좀 충격','틱톡','틱톡 프롬','좀 폐쇄적','바쁜와중','정기컨텐츠','한다는게','생방들','우리','보고','생각','메모','유튜브','신젤리','클립','타임','타임 코드','타임 코드 메모','타임 코드 메모하는 언냐들','타임 코드 메모하는 언냐들 좀','신입','신입들어오면 잘난체좀','컨텐츠','컨텐츠 끝나면 게시글','컨텐츠 끝나면 게시글 하나','포인트','소수자','호캉스','호캉스 타임코드','호캉스 타임코드 메모','만해','존나','방도','방도 개웃겼긔','왤케','왤케 촉촉','왤케 촉촉 아련','편집자','좆무위키','좆무위키 정독','좆무위키 정독시키면 되긔','레즈대상','필요','유튭','유튭 브이로그','속옷얘기','진짜 답변','진짜 답변 생각','못했긔','금주데','편집포인트','좆돌','좆돌 예능','좆돌 예능 예고편','댕이','에프','에프더','누가','웃기','가위','젤사원들','젤사원들의 새로운 회의','시작','군하','바지 내리는 질문','무슨빵','참고','프롬','프롬 선예매','프롬 선예매 고민','어차피','어차피 우리','얘기','얘기하는 웃긴 부분','이쁜속옷','이쁜속옷 막','이쁜속옷 막입는속옷','있으신가보긔','의논','의논해주는 것','ai로','ai로 숏폼','ai로 숏폼 제작','ai로 숏폼 제작하는 거도','ai로 숏폼 제작하는 거도 있눈덴','어필','어필되는 속옷','어필되는 속옷 기준','우리 의견','별거','별거 아닐 수','중요한건 팬','그냥','그냥 걱정','백허그','백허그 유or','앞허그중','우린','우린 응원조','우린 응원조 정도긔','부심','소찌키 혐애','소찌키 혐애 잘','어필얘기','타임코드','타임코드 될거같긔','뭔가','뭔가 더','뭔가 더 좋은것','시언','이때','이때 울면','걱정했긔','꿀떨어질 때','꿀떨어질 때 우리','꿀떨어질 때 우리 무슨말','시기','12월','한달이','시간있긔','청주여자교도소','주인님','다시보기','다시보기 싱크','방 끊어가길 잘','뒷부분','뒷부분 존나','싱크','싱크 뭐','오늘거','오늘거 싱크','오늘거 싱크 안','오늘거 싱크 안맞는건 문화유산','오늘거 싱크 안맞는건 문화유산 훼손','마지막쯤','마지막쯤 밀은 아예','복습 뭐','화녹','손목','전완근','이번','이번 팬','이번 팬미는 포옹','오늘거 복습','오늘거 복습 못','허그','와락','방 시험기간','봊친','이런 건','이런 건 안','오늘 뭔가','오늘 뭔가 믿음직','그 와중','스폰지밥','커비','커비 가져가면 안대','댕 약간','댕 약간 촉촉','댕될때 놈 좋긔','댕 오늘','겉부속촉','임명','언니 오늘','일찍','일찍 푹','오늘 유독','오늘 유독 두부','사실','사실 다음주','여러가지','방 복습중','배웅','배웅할때','배웅할때 산타걸','밍숭짭댕','본인','무쌍','무쌍같을 때','금주','금주의 데이트','금주의 데이트 전부','금주의 데이트 전부 실시간','벌써','금순언니','다음주','다음주 일정','일 처리','잡댕','영상','영상 싱크','영상 싱크 겨우','왜케','뽑기','질문','저런','대답','가지','순자아','촉촉','훈훈','아디','지랄','현실','학교','시험','기간','피곤','의젓','최종','가오','핏줄','좋드','눈빛','막판','괴고','댓글','머리','금순','미녀','지원','봊돌','소신','중년','저씨들','여자','어른','충격','폐쇄적','와중','정기','코드','언냐들','체좀','게시','하나','개웃겼긔','아련','좆무','위키','정독','되긔','레즈','대상','브이','로그','속옷','답변','편집','예능','예고편','사원들','회의','예매','고민','부분','가보','ai','숏폼','제작','거도','있눈덴','기준','의견','걱정','or','허그중','응원조','정도','혐애','될거같긔','울면','했긔','무슨말','시간','있긔','청주','여자교도소','다시','보기','문화유산','훼손','아예','전완','포옹','믿음','안대','약간','좋긔','부속','유독','두부','복습중','산타','데이트','전부','실시간','순언니','일정','처리','겨우'
    ]
    
    # 필터링 임계값 (0~1)
    MATCH_THRESHOLD = 0.02  # 2% 이상 매칭되면 포함
    # =============================================
    
    # 폴더 존재 확인
    if not os.path.exists(SHORTS_FOLDER):
        print(f"❌ 폴더를 찾을 수 없습니다: {SHORTS_FOLDER}")
        print(f"💡 '{SHORTS_FOLDER}' 폴더를 생성하고 쇼츠 영상들을 넣어주세요")
        return
    
    # ffmpeg 확인
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg/ffprobe가 설치되어 있지 않습니다!")
        return
    
    print("🎬 쇼츠 배치 분석 시스템")
    print("=" * 80)
    print(f"📁 입력 폴더: {SHORTS_FOLDER}")
    print(f"📁 출력 폴더: {OUTPUT_DIR}")
    print(f"🎯 매칭 임계값: {MATCH_THRESHOLD:.0%}")
    print(f"🔑 커뮤니티 키워드: {len(COMMUNITY_KEYWORDS)}개")
    print("=" * 80)
    
    # 배치 처리기 초기화
    processor = ShortsBatchProcessor(
        shorts_folder=SHORTS_FOLDER,
        output_dir=OUTPUT_DIR,
        community_keywords=COMMUNITY_KEYWORDS,
        match_threshold=MATCH_THRESHOLD
    )
    
    # 1. 모든 쇼츠 분석
    all_results = processor.process_all_shorts()
    
    if not all_results:
        print("\n❌ 분석할 수 있는 쇼츠가 없습니다")
        return
    
    # 2. 주제 모델링
    all_results = processor.add_topic_modeling(all_results)
    
    # 3. 제목 생성
    all_results = processor.generate_titles(all_results)
    
    # 4. 필터링 및 파일 복사
    filtered_results = processor.filter_and_copy_shorts(all_results)
    
    # 5. 리포트 저장
    processor.save_reports(all_results, filtered_results)
    
    # 완료!
    print("\n" + "=" * 80)
    print("🎉 완료!")
    print("=" * 80)
    print(f"📁 저장 위치: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📊 분석된 쇼츠: {len(all_results)}개")
    print(f"✅ 필터링 통과: {len(filtered_results)}개")
    print(f"📈 필터링 비율: {len(filtered_results)/max(len(all_results), 1):.1%}")
    
    if filtered_results:
        print(f"\n🏆 Top 5 쇼츠 (매칭 점수순):")
        top_5 = sorted(filtered_results, key=lambda x: x['match_score'], reverse=True)[:5]
        for i, result in enumerate(top_5, 1):
            print(f"  {i}. {result.get('title', 'N/A')}")
            print(f"     매칭: {result['match_score']:.0%} | 감정: {result['sentiment']}")
    
    print("\n💡 생성된 파일:")
    print(f"  - {len(filtered_results)}개의 필터링된 쇼츠 영상")
    print(f"  - analysis_report.json (상세 분석 결과)")
    print(f"  - report.txt (읽기 쉬운 리포트)")
    print(f"  - summary.csv (엑셀용 요약)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        from faster_whisper import WhisperModel
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
    except ImportError:
        print("❌ 필요한 라이브러리를 설치해주세요:")
        print("\npip install faster-whisper scikit-learn")
        print("pip install kiwipiepy  # 한국어 형태소 분석기 (권장)")
        exit(1)
    
    main()