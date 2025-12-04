"""Pinecone 검색 파이프라인"""
from typing import List, Dict, Any, Optional
import logging
import time

from ..insights.metadata_extractor import MetadataExtractor
from .metadata_filter_extractor import MetadataFilterExtractor
from ..insights.category_classifier import CategoryClassifier
from ..insights.text_generator import CategoryTextGenerator
from .embedding_generator import EmbeddingGenerator
from .pinecone_searcher import PineconePanelSearcher
from .pinecone_result_filter import PineconeResultFilter

logger = logging.getLogger(__name__)


class PanelSearchPipeline:
    """전체 검색 파이프라인 (Pinecone + LLM 기반 메타데이터 필터)"""

    def __init__(
        self,
        pinecone_api_key: str,
        pinecone_index_name: str,
        category_config: Dict[str, Any],
        anthropic_api_key: str,
        openai_api_key: str
    ):
        """
        Args:
            pinecone_api_key: Pinecone API 키
            pinecone_index_name: Pinecone 인덱스 이름
            category_config: 카테고리 설정 딕셔너리
            anthropic_api_key: Anthropic API 키
            openai_api_key: OpenAI API 키
        """
        self.metadata_extractor = MetadataExtractor(anthropic_api_key)
        self.filter_extractor = MetadataFilterExtractor(anthropic_api_key)
        self.category_classifier = CategoryClassifier(category_config, anthropic_api_key)
        self.text_generator = CategoryTextGenerator(anthropic_api_key)
        self.embedding_generator = EmbeddingGenerator(openai_api_key)
        self.searcher = PineconePanelSearcher(pinecone_api_key, pinecone_index_name, category_config)
        self.result_filter = PineconeResultFilter(self.searcher)

    def search(self, query: str, top_k: int = None, external_filters: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
        """
        자연어 쿼리로 패널 검색

        Args:
            query: 검색 쿼리 (예: "서울 20대 남자")
            top_k: 반환할 패널 수 (None이면 조건 만족하는 전체 반환)
            external_filters: 외부 필터 (카테고리별 Pinecone 필터)
                예: {"기본정보": {"지역": {"$in": ["서울"]}}, "직업소득": {...}}

        Returns:
            mb_sn 리스트
        """
        start_time = time.time()

        # 빈 쿼리이고 외부 필터만 있는 경우
        if (not query or not query.strip()) and external_filters:
            logger.info("[검색] 빈 쿼리, 외부 필터만으로 검색")
            # 필터만으로 검색 진행 (임베딩 생성 불필요)
            metadata = {}
            # 외부 필터의 카테고리로 classified 초기화 (필터만으로 검색 가능하도록)
            classified = {cat: {} for cat in external_filters.keys()}
            final_count = top_k  # top_k가 None이면 전체 반환
            logger.info(f"[검색] 외부 필터 카테고리: {list(classified.keys())}")
        else:
            # 1단계: 메타데이터 추출
            step_start = time.time()
            logger.info("[1단계] 메타데이터 추출 시작")
            try:
                if query and query.strip():
                    metadata = self.metadata_extractor.extract(query)
                else:
                    metadata = {}
                step_time = time.time() - step_start
                logger.info(f"[1단계 완료] 메타데이터 추출: {step_time:.2f}초, 결과: {metadata}")
            except Exception as e:
                logger.warning(f"[1단계 경고] 메타데이터 추출 실패 (계속 진행): {e}")
                metadata = {}  # 빈 메타데이터로 계속 진행
            
            # 메타데이터 추출 실패 시 필터 폴백 처리
            if not metadata:
                if external_filters:
                    # 필터가 있으면 필터만으로 검색 진행
                    logger.warning("[경고] 메타데이터 추출 실패, 필터만으로 검색 진행")
                    # 필터만 검색 모드로 전환
                    classified = {cat: {} for cat in external_filters.keys()}
                    final_count = top_k  # top_k가 None이면 전체 반환
                    logger.info(f"[검색] 외부 필터 카테고리: {list(classified.keys())}")
                else:
                    # 필터도 없으면 검색 불가
                    logger.error("[ERROR] 메타데이터 추출 실패 - 빈 메타데이터 반환, 필터도 없음")
                    return []
            
            # top_k 결정: query에서 추출 또는 파라미터 사용
            final_count = None  # 기본값: None (전체 반환)
            
            if top_k is not None:
                # 파라미터로 명시적 전달
                final_count = top_k
                logger.info(f"\n[인원수] 파라미터로 {final_count}명 지정됨")
            else:
                # metadata에서 인원수 추출 시도
                extracted_count = metadata.get("인원수")
                if extracted_count and isinstance(extracted_count, int) and extracted_count > 0:
                    final_count = extracted_count
                    logger.info(f"\n[인원수] 쿼리에서 {final_count}명 추출됨")
                else:
                    # 명수 미명시 → None으로 유지 (필터링 후 남은 모든 후보 반환)
                    final_count = None
                    logger.info(f"\n[인원수] 쿼리에서 인원수 미명시 → 조건 만족하는 전체 패널 반환")

            # 인원수 키 제거 (검색 조건이 아닌 결과 개수 지정용)
            if "인원수" in metadata:
                metadata.pop("인원수")
                logger.info(f"[메타데이터 정리] '인원수' 키 제거 완료")

        # 2단계: 카테고리 분류
        step_start = time.time()
        logger.info(f"[2단계] 카테고리 분류 시작 (메타데이터: {metadata})")
        try:
            if metadata:
                classified = self.category_classifier.classify(metadata)
                if not classified:
                    logger.warning(f"[2단계 경고] 카테고리 분류 결과가 비어있음 (메타데이터: {metadata})")
                    # rule-based 폴백이 이미 시도되었을 수 있으므로, 메타데이터가 있으면 외부 필터와 병합 시도
                    if external_filters:
                        logger.info(f"[2단계] 외부 필터와 병합 시도: {external_filters}")
                        classified = {cat: {} for cat in external_filters.keys()}
            else:
                # 메타데이터가 없으면 외부 필터의 카테고리 사용
                if external_filters:
                    classified = {cat: {} for cat in external_filters.keys()}
                    logger.info(f"[2단계] 메타데이터 없음, 외부 필터 카테고리 사용: {list(classified.keys())}")
                else:
                    classified = {}
                    logger.warning(f"[2단계] 메타데이터와 외부 필터 모두 없음")
            step_time = time.time() - step_start
            logger.info(f"[2단계 완료] 카테고리 분류: {step_time:.2f}초, 결과: {classified}")
        except Exception as e:
            logger.warning(f"[2단계 경고] 카테고리 분류 실패 (계속 진행): {e}", exc_info=True)
            classified = {}  # 빈 분류로 계속 진행
        
        if not classified:
            # 인원수만 있고 다른 메타데이터가 없는 경우: 쿼리 텍스트를 직접 임베딩해서 검색
            if final_count is not None and not metadata and not external_filters:
                logger.info(f"[Fallback] 인원수만 지정됨 ({final_count}명), 쿼리 텍스트 직접 검색으로 폴백")
                # 쿼리 텍스트를 직접 임베딩해서 검색
                try:
                    # 모든 카테고리에서 검색 (기본정보 카테고리 사용)
                    category_config = self.category_classifier.category_config
                    default_category = "기본정보"  # 기본 카테고리
                    
                    # 쿼리 텍스트를 직접 임베딩
                    query_text_embedding = self.embedding_generator.generate(query)
                    
                    # Pinecone에서 검색 (필터 없이)
                    searcher = self.searcher
                    results = searcher.search_by_category(
                        query_text_embedding,
                        default_category,
                        top_k=final_count * 10 if final_count else 10000,  # 여유있게 검색
                        filter_mb_sns=None,
                        metadata_filter=None
                    )
                    
                    # 유사도 점수 기준으로 정렬 (내림차순) - Pinecone이 이미 정렬하지만 확실히 하기 위해
                    # Pinecone의 query()는 이미 유사도 점수 기준 내림차순으로 정렬된 결과를 반환하지만,
                    # 명시적으로 정렬하여 상위 유사도 패널만 반환하도록 보장
                    sorted_results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
                    
                    # 최종 개수만큼 반환 (상위 유사도 패널만)
                    final_results = sorted_results[:final_count] if final_count else sorted_results
                    mb_sns_with_scores = [{"mb_sn": r["mb_sn"], "score": r["score"]} for r in final_results]
                    
                    # 디버그: 상위 5개 점수 로깅
                    if final_results:
                        top_scores = [r["score"] for r in final_results[:5]]
                        logger.info(f"[Fallback] 쿼리 텍스트 직접 검색 완료: {len(mb_sns_with_scores)}개 패널 (상위 5개 점수: {top_scores})")
                    else:
                        logger.info(f"[Fallback] 쿼리 텍스트 직접 검색 완료: {len(mb_sns_with_scores)}개 패널")
                    return mb_sns_with_scores
                except Exception as e:
                    logger.error(f"[Fallback] 쿼리 텍스트 직접 검색 실패: {e}", exc_info=True)
                    return []
            else:
                logger.error("[ERROR] 카테고리 분류 실패 - 빈 분류 결과")
                return []

        # 2.5단계: 카테고리별 메타데이터 필터 추출 및 정규화
        step_start = time.time()
        logger.info("[2.5단계] 카테고리별 메타데이터 필터 추출 시작")
        category_filters = {}
        
        # 외부 필터와 쿼리 필터 중 하나만 사용 (병합하지 않음)
        # 쿼리에서 추출한 필터 확인 (비교용)
        query_extracted_filters = {}
        for category in classified.keys():
            cat_filter = self.filter_extractor.extract_filters(metadata, category)
            if cat_filter:
                query_extracted_filters[category] = cat_filter
        
        if external_filters:
            # 외부 필터가 있으면 외부 필터만 사용 (쿼리 필터 무시)
            logger.info(f"[2.5단계] 외부 필터만 사용 (쿼리 필터 무시)")
            category_filters.update(external_filters)
        else:
            # 외부 필터가 없으면 쿼리에서 추출한 필터만 사용
            logger.info(f"[2.5단계] 쿼리 필터만 사용 (외부 필터 없음)")
            category_filters.update(query_extracted_filters)
        
        step_time = time.time() - step_start
        logger.info(f"[2.5단계 완료] 필터 추출: {step_time:.2f}초, 결과: {category_filters}")

        # 필터만 검색하는 경우 (빈 쿼리 + 외부 필터만): 임베딩 생성 생략하고 바로 필터 검색
        is_filter_only_search = (not query or not query.strip()) and external_filters and not metadata
        
        if is_filter_only_search:
            logger.info("[검색] 필터만 검색 모드 - 임베딩 생성 생략, 메타데이터 필터만으로 검색")
            
            # 랜덤 벡터 생성 (Pinecone 검색에 필요하지만 유사도는 무시)
            import numpy as np
            dimension = 1536  # OpenAI text-embedding-3-small embedding dimension
            random_vector = np.random.rand(dimension).astype(np.float32).tolist()
            norm = np.linalg.norm(random_vector)
            if norm > 0:
                random_vector = (np.array(random_vector) / norm).tolist()
            
            # 각 카테고리별로 동일한 랜덤 벡터 사용 (유사도는 무시하고 필터만 적용)
            embeddings = {}
            category_order = list(category_filters.keys())
            for category in category_order:
                embeddings[category] = random_vector
            
            logger.info(f"[검색] 랜덤 벡터 생성 완료, 카테고리: {category_order}")
        else:
            # 3단계: 자연어 텍스트 생성 (순차 처리)
            step_start = time.time()
            logger.info("[3단계] 자연어 텍스트 생성 시작 (순차 처리)")
            texts = {}
            
            if classified:
                # 순차 처리로 변경 (카테고리 순서 보장)
                for category, items in classified.items():
                    try:
                        # full_metadata_dict=metadata로 키워드 인자 전달
                        text = self.text_generator.generate(category, items, full_metadata_dict=metadata)
                        if text and text.strip():
                            texts[category] = text
                        else:
                            logger.warning(f"[WARN] 텍스트 생성 결과가 비어있음 ({category})")
                    except Exception as e:
                        logger.error(f"[ERROR] 텍스트 생성 중 예외 발생 ({category}): {e}", exc_info=True)
            
            step_time = time.time() - step_start
            logger.info(f"[3단계 완료] 텍스트 생성: {step_time:.2f}초, 카테고리 수: {len(texts)}")

            # 4단계: 임베딩 생성
            step_start = time.time()
            logger.info("[4단계] 임베딩 생성 시작")
            
            try:
                if texts:
                    embeddings = self.embedding_generator.generate(texts)
                else:
                    # 텍스트가 없으면 랜덤 벡터 사용
                    logger.warning(f"[4단계] 텍스트 없음, 랜덤 벡터 사용 (유사도 기반 검색 불가)")
                    logger.warning(f"[4단계] classified 카테고리: {list(classified.keys()) if classified else []}")
                    import numpy as np
                    dimension = 1536
                    random_vector = np.random.rand(dimension).astype(np.float32).tolist()
                    norm = np.linalg.norm(random_vector)
                    if norm > 0:
                        random_vector = (np.array(random_vector) / norm).tolist()
                    
                    embeddings = {}
                    for category in classified.keys() if classified else []:
                        embeddings[category] = random_vector
                    
                    logger.warning(f"[4단계] 랜덤 벡터로 검색 (필터만 적용, 유사도 무시)")
                
                step_time = time.time() - step_start
                logger.info(f"[4단계 완료] 임베딩 생성: {step_time:.2f}초, 카테고리 수: {len(embeddings)}")
            except Exception as e:
                logger.error(f"[4단계 에러] 임베딩 생성 실패: {e}", exc_info=True)
                logger.warning(f"[4단계] 임베딩 생성 실패로 빈 embeddings 사용")
                embeddings = {}
            
            if not embeddings:
                logger.warning("[경고] 임베딩이 비어있음, 검색 불가")
                return []
            
           # classified의 키 순서 사용 (카테고리 순서 보장)
            category_order = list(classified.keys()) if classified else list(embeddings.keys())

        # 5단계: 단계적 필터링 검색
        step_start = time.time()
        logger.info("[5단계] 단계적 필터링 검색 시작")

        final_results = self.result_filter.filter_by_categories(
            embeddings=embeddings,
            category_order=category_order,
            final_count=final_count,
            topic_filters=category_filters
        )
        step_time = time.time() - step_start
        logger.info(f"[5단계 완료] 단계적 필터링: {step_time:.2f}초, 최종 결과: {len(final_results)}개")

        total_time = time.time() - start_time
        logger.info(f"[검색 완료] 총 소요 시간: {total_time:.2f}초, 결과: {len(final_results)}개 패널")

        # mb_sn 리스트와 score 맵 반환
        mb_sns = [r["mb_sn"] for r in final_results]
        score_map = {r["mb_sn"]: r["score"] for r in final_results}
        
        return {"mb_sns": mb_sns, "scores": score_map}

