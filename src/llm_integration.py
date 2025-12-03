"""
LLM integration module for labeling, response generation, explanations, and data augmentation.
"""
import time
import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
from google import genai
import pandas as pd
from . import config

logger = logging.getLogger(__name__)


@dataclass
class LabeledExample:
    text: str
    intent: str
    severity: int


class LLMIntegration:
    """Handles all LLM interactions for the customer support system."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_MODEL_NAME
        self.client = genai.Client(api_key=self.api_key)
        self.quota_exhausted = False

    def _is_quota_error(self, message: str) -> bool:
        """Heuristic to detect quota/rate-limit errors without depending on error types."""
        if not message:
            return False
        msg = message.lower()
        return "resource_exhausted" in msg or "quota" in msg or "429" in msg

    def _mark_quota_exhausted(self, message: str) -> None:
        """Remember quota exhaustion to short-circuit further calls."""
        self.quota_exhausted = True
        logger.warning("Gemini quota exhausted; skipping further LLM calls. Details: %s", message[:160])
    
    def build_label_prompt(self, tweet_text: str) -> str:
        """Build prompt for intent and severity labeling."""
        return f"""
You are labeling a customer support tweet.

Tweet:
\"\"\"{tweet_text}\"\"\"

Step 1: Decide the main complaint category (intent) as a short snake_case label. Examples:
- billing_issue
- technical_issue
- account_access
- delivery_or_service_delay
- refund_or_compensation
- general_question
- other

Step 2: Decide a severity score from 0 to 3:
0 = informational / no problem
1 = mild complaint
2 = serious problem but not life-critical
3 = critical / urgent / very angry / repeated failures

Now OUTPUT ONLY TWO LINES and NOTHING ELSE:

Line 1: the intent label (just the label, no explanation)
Line 2: the severity as an integer between 0 and 3

Example output:
billing_issue
2
"""
    
    def parse_label_response(self, text: str) -> Optional[Tuple[str, int]]:
        """Parse the 2-line format from Gemini into (intent, severity)."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return None
        
        # First non-empty line is intent
        intent_raw = lines[0]
        intent_raw = intent_raw.lower().replace("intent:", "").strip()
        if not intent_raw:
            intent_raw = "other"
        
        # Find severity digit 0–3 on subsequent lines
        severity = 1
        for l in lines[1:]:
            digits = "".join(ch for ch in l if ch.isdigit())
            if digits:
                try:
                    severity = int(digits)
                    break
                except ValueError:
                    continue
        
        severity = max(0, min(3, severity))
        return intent_raw, severity
    
    def call_gemini_for_labels(self, text: str) -> Union[Optional[LabeledExample], str]:
        """
        Call Gemini to get intent + severity.
        Returns:
          - LabeledExample on success
          - None on normal error
          - "QUOTA_EXCEEDED" string if hitting rate/quota limits
        """
        if self.quota_exhausted:
            return "QUOTA_EXCEEDED"

        prompt = self.build_label_prompt(text)
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            raw = resp.text.strip()
            parsed = self.parse_label_response(raw)
            if parsed is None:
                logger.warning(f"Could not parse label response: {repr(raw[:120])}...")
                return None
            
            intent, severity = parsed
            return LabeledExample(text=text, intent=intent, severity=severity)
        
        except Exception as e:
            msg = str(e)
            if self._is_quota_error(msg):
                self._mark_quota_exhausted(msg)
                return "QUOTA_EXCEEDED"
            logger.error(f"Error calling Gemini for labels: {msg[:200]}...", exc_info=True)
            return None
    
    def build_reply_prompt(self, tweet_text: str, intent: str, severity: int) -> str:
        """Build prompt for generating support replies."""
        return f"""
You are a professional customer support agent.

Customer tweet:
\"\"\"{tweet_text}\"\"\"

Predicted intent: {intent}
Predicted severity (0-3): {severity}

Write a short, empathetic, and helpful reply (max 3 sentences) that:
- acknowledges the issue,
- gives a clear next step,
- stays polite and professional.
No hashtags, no emojis. Just the reply text.
"""
    
    def generate_support_reply(self, tweet_text: str, intent: str, severity: int) -> str:
        """Generate a support reply using LLM."""
        if self.quota_exhausted:
            return "LLM quota exceeded; cannot generate a reply right now."

        prompt = self.build_reply_prompt(tweet_text, intent, severity)
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return resp.text.strip()
        except Exception as e:
            msg = str(e)
            if self._is_quota_error(msg):
                self._mark_quota_exhausted(msg)
                return "LLM quota exceeded; cannot generate a reply right now."
            logger.error(f"Error generating support reply: {msg[:200]}...", exc_info=True)
            return "Sorry, something went wrong while generating a reply."
    
    def build_explanation_prompt(self, tweet_text: str, predicted_intent: str, predicted_severity: int) -> str:
        """Build prompt for generating classification explanations."""
        return f"""
You are explaining a machine learning model's prediction.

Customer tweet:
\"\"\"{tweet_text}\"\"\"

The model predicted:
- Intent category: {predicted_intent}
- Severity level: {predicted_severity} (0=informational, 1=mild, 2=serious, 3=critical)

Write a brief explanation (2-3 sentences) explaining why the model classified this tweet as "{predicted_intent}" with severity {predicted_severity}. 
Focus on the key words, phrases, or patterns in the tweet that led to this classification.
Be clear and concise.
"""
    
    def generate_explanation(self, tweet_text: str, predicted_intent: str, predicted_severity: int) -> str:
        """
        Generate explanation for classifier predictions (2-3 sentences as per requirements).
        """
        if self.quota_exhausted:
            return "LLM quota exceeded; cannot generate an explanation right now."

        prompt = self.build_explanation_prompt(tweet_text, predicted_intent, predicted_severity)
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return resp.text.strip()
        except Exception as e:
            msg = str(e)
            if self._is_quota_error(msg):
                self._mark_quota_exhausted(msg)
                return "LLM quota exceeded; cannot generate an explanation right now."
            logger.error(f"Error generating explanation: {msg[:200]}...", exc_info=True)
            return "Unable to generate explanation at this time."
    
    def build_augmentation_prompt(self, intent: str, n_examples: int = 3) -> str:
        """Build prompt for generating synthetic complaint examples."""
        return f"""
Generate {n_examples} realistic customer support tweets/complaints for the category: {intent}

Requirements:
- Each tweet should be realistic and natural (like real social media complaints)
- Vary the wording, tone, and specific issues
- Keep tweets concise (under 280 characters)
- Include common typos, casual language, and social media style
- Each tweet should clearly belong to the "{intent}" category

Output format: One tweet per line, no numbering or bullets.
"""
    
    def generate_augmented_samples(self, intent: str, n_samples: int = 5) -> List[str]:
        """
        Generate synthetic complaint examples for data augmentation.
        Used to balance underrepresented classes.
        """
        if self.quota_exhausted:
            return []

        prompt = self.build_augmentation_prompt(intent, n_examples=n_samples)
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            # Parse the response into individual tweets
            lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
            # Filter out any lines that look like formatting (numbers, bullets, etc.)
            samples = []
            for line in lines:
                # Remove common formatting prefixes
                line = re.sub(r'^[\d\.\-\*\)]+\s*', '', line)
                if len(line) > 10:  # Only keep substantial lines
                    samples.append(line)
            
            return samples[:n_samples]  # Return up to n_samples
        except Exception as e:
            msg = str(e)
            if self._is_quota_error(msg):
                self._mark_quota_exhausted(msg)
                return []
            logger.error(f"Error generating augmented samples for {intent}: {msg[:200]}...", exc_info=True)
            return []
    
    def augment_dataset(self, labeled_df, min_samples_per_class: int = 5) -> pd.DataFrame:
        """
        Augment dataset by generating synthetic samples for underrepresented classes.
        
        Args:
            labeled_df: DataFrame with 'text', 'intent', 'severity' columns
            min_samples_per_class: Minimum samples desired per intent class
        
        Returns:
            Augmented DataFrame
        """
        import pandas as pd
        
        augmented_rows = []
        intent_counts = labeled_df['intent'].value_counts()
        
        classes_to_augment = {intent: count for intent, count in intent_counts.items() 
                             if count < min_samples_per_class}
        
        if not classes_to_augment:
            logger.info("No classes need augmentation")
            return labeled_df
        
        logger.info(f"Augmenting {len(classes_to_augment)} underrepresented classes...")
        
        quota_exceeded = False
        for intent, count in classes_to_augment.items():
            if quota_exceeded:
                break
            if self.quota_exhausted:
                break
                
            needed = min_samples_per_class - count
            logger.info(f"Augmenting {intent}: generating {needed} synthetic samples...")
            
            synthetic_texts = self.generate_augmented_samples(intent, n_samples=needed)
            
            if not synthetic_texts:
                logger.warning(f"No synthetic samples generated for {intent}")
                continue
            
            for text in synthetic_texts:
                if not text or len(text.strip()) == 0:
                    continue
                    
                # Label the synthetic sample
                labeled = self.call_gemini_for_labels(text)
                if isinstance(labeled, LabeledExample):
                    augmented_rows.append({
                        'text': text,
                        'intent': labeled.intent,
                        'severity': labeled.severity
                    })
                elif labeled == "QUOTA_EXCEEDED":
                    logger.warning("Hit API quota during augmentation, stopping")
                    quota_exceeded = True
                    break
                time.sleep(config.API_DELAY)
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            logger.info(f"Added {len(augmented_rows)} augmented samples")
            return pd.concat([labeled_df, augmented_df], ignore_index=True)
        
        return labeled_df
