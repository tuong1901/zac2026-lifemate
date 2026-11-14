"""High-confidence domain router over structured intent chunks."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever


class StructuredDomainRouter:
    """Predict domains, then retrieve a workflow only inside those domains."""

    def __init__(
        self,
        retriever: BM25ExemplarRetriever,
        threshold: float = -0.4,
        candidate_pool: int = 50,
        workflow_pool: int = 3,
        vote_weight: float = 0.3,
    ):
        self.retriever = retriever
        self.threshold = threshold
        self.candidate_pool = candidate_pool
        self.workflow_pool = workflow_pool
        self.vote_weight = vote_weight
        self.vectorizer = None
        self.classifier = None
        self.labeler = None

    @staticmethod
    def _workflow(record: Dict[str, Any]) -> Tuple[str, ...]:
        return tuple(call.get("name", "") for call in record.get("tool_calls", []))

    def fit(self) -> "StructuredDomainRouter":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.svm import LinearSVC

        texts = [record.get("audio_text", "") for record in self.retriever.corpus]
        domains = [sorted(record.get("called_apps", set())) for record in self.retriever.corpus]
        self.labeler = MultiLabelBinarizer()
        labels = self.labeler.fit_transform(domains)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        features = self.vectorizer.fit_transform(texts)
        self.classifier = OneVsRestClassifier(LinearSVC(C=0.5))
        self.classifier.fit(features, labels)
        return self

    def predict_domains(self, query_text: str) -> Set[str]:
        if self.vectorizer is None or self.classifier is None or self.labeler is None:
            raise RuntimeError("StructuredDomainRouter.fit() must be called first")
        scores = self.classifier.decision_function(self.vectorizer.transform([query_text]))
        row = scores[0]
        return {
            self.labeler.classes_[index]
            for index, score in enumerate(row)
            if score >= self.threshold
        }

    def retrieve(
        self,
        query_text: str,
        installed_apps: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        candidates = self.retriever.retrieve(
            query_text=query_text,
            installed_apps=installed_apps,
            top_k=self.candidate_pool,
        )
        if not candidates:
            return []

        predicted_domains = self.predict_domains(query_text)
        matches = [
            record for record in candidates
            if set(record.get("called_apps", set())) == predicted_domains
        ][:self.workflow_pool]
        if not matches:
            return [candidates[0]]

        grouped: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
        for record in matches:
            grouped[self._workflow(record)].append(float(record.get("bm25_score", 0.0)))
        winner = max(
            grouped,
            key=lambda workflow: grouped[workflow][0] + self.vote_weight * sum(grouped[workflow][1:]),
        )
        # OOS is handled by the dedicated gatekeeper. A weak empty-domain
        # prediction must not erase a concrete BM25 workflow.
        if not winner and self._workflow(candidates[0]):
            return [candidates[0]]
        return [next(record for record in matches if self._workflow(record) == winner)]
