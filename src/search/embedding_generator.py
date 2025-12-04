"""임베딩 생성기"""
from typing import Dict, List
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """OpenAI text-embedding-3-small로 임베딩 생성 (순차 처리)"""

    def __init__(self, api_key: str):
        """
        Args:
            api_key: OpenAI API 키
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"

    def generate(self, texts: Dict[str, str]) -> Dict[str, List[float]]:
        """
        카테고리별 임베딩 생성 (순차 처리)
        
        Args:
            texts: 카테고리별 텍스트 딕셔너리
            
        Returns:
            카테고리별 임베딩 딕셔너리
        """
        result = {}
        
        # 순차 처리로 변경 (카테고리 순서 보장)
        for category, text in texts.items():
            if not text:
                continue
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                embedding = response.data[0].embedding
                result[category] = embedding
                logger.info(f"[{category}] 임베딩 생성 완료")
            except Exception as e:
                logger.error(f"[{category}] 임베딩 생성 실패: {e}")

        return result

