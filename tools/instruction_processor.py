# tools/instruction_processor.py
from typing import Dict, Tuple
from openai import OpenAI
import re

class InstructionProcessor:
    def __init__(
        self, 
        model_api_base: str = "http://localhost:8011/v1", 
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
        confidence_threshold: float = 80 
    ):
        """
        Initialize instruction processor
        """
        self.config = {
            "api_key": "EMPTY",
            "api_base": model_api_base,
            "model_name": model_name,
            "temperature": 0.1,
            "timeout": 60
        }
        self.confidence_threshold = confidence_threshold
        self.client = None 

    def __enter__(self):
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["api_base"]
        )
        return self

    def __exit__(self, *args):
        if self.client:
            self.client.close()
        self.client = None

    def _classify_with_qwen(self, instruction: str) -> Tuple[str, str, float]:
        """
        Robust classification using Regex search instead of strict split
        """
        if not self.client:
            raise RuntimeError("InstructionProcessor must be used with 'with' statement")

        prompt = f"""
        Task: Classify this aerial image detection instruction.
        
        1. Intent: Explicit, Implicit, or Ambiguous?
        2. Difficulty: Simple or Complex?
        3. Confidence: 0-100?

        Instruction: "{instruction}"
        
        Output format: "Intent, Difficulty, Score" (Just the values)
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50, # 稍微给多点空间
                timeout=self.config["timeout"]
            )
            result = response.choices[0].message.content.strip()
            
            # --- DEBUG: 打印出来看看模型到底说了啥 (这一行很重要) ---
            print(f"[Classify Raw Output]: {result}") 
            # ----------------------------------------------------

            # 1. 更加鲁棒的解析方法：转小写后找关键词
            text_lower = result.lower()

            # 提取 Intent
            if "explicit" in text_lower:
                intent = "Explicit"
            elif "implicit" in text_lower:
                intent = "Implicit"
            else:
                intent = "Ambiguous" # 默认值

            # 提取 Difficulty
            if "simple" in text_lower:
                difficulty = "Simple"
            else:
                difficulty = "Complex" # 默认值，倾向于认为是复杂的

            # 提取 Confidence (找字符串里的最后一个数字)
            # 逻辑：通常分数在最后。如果没有数字，给 0.0
            numbers = re.findall(r'\d+', result)
            if numbers:
                confidence = float(numbers[-1]) # 取最后一个数字
                confidence = max(0.0, min(100.0, confidence)) # 限制在 0-100
            else:
                print(f"[Warning] No score found in: {result}")
                confidence = 0.0

            return intent, difficulty, confidence

        except Exception as e:
            print(f"[Qwen classification failed] Error: {str(e)}")
            return "Ambiguous", "Complex", 0.0

    def decompose_instruction(self, raw_instruction: str) -> str:
        if not self.client:
            raise RuntimeError("InstructionProcessor must be used with 'with' statement")

        prompt = f"""
        Task: Decompose the aerial image detection instruction into "Targets" and "Environment".
        Format: "Targets: [list]; Environment: [desc]"
        Rules: Use "none" if a part does not exist.
        Original instruction: {raw_instruction}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config["temperature"],
                max_tokens=100,
                timeout=self.config["timeout"]
            )
            processed_result = response.choices[0].message.content.strip()
            if "Targets: none" in processed_result.lower() and "environment: none" in processed_result.lower():
                return raw_instruction
            return processed_result
        except Exception as e:
            print(f"[Instruction decomposition failed] Error: {str(e)}")
            return raw_instruction

    def process(self, instruction: str) -> Tuple[Dict, str]:
        """Complete processing flow: Classification -> Routing -> Processing"""
        # 1. Classification
        intent, difficulty, confidence = self._classify_with_qwen(instruction)
        
        process_result = {
            "raw_instruction": instruction,
            "intent": intent,
            "difficulty": difficulty,
            "confidence_score": confidence,
            "confidence_threshold": self.confidence_threshold,
            "classifier": "Qwen2.5",
            "decomposed": False
        }
        print(process_result)

        # 2. Logic with Safe Defaults
        # 初始化 final_instruction 为原始指令，防止 crash
        final_instruction = instruction

        # 路由逻辑：
        # 情况A：置信度太低 -> 不分解
        if confidence < self.confidence_threshold:
            process_result["decomposed"] = False
            process_result["decomposition_reason"] = "Low confidence"
        
        # 情况B：指令非常明确且简单 (Explicit + Simple) -> 不分解
        elif intent == "Explicit" and difficulty == "Simple":
            process_result["decomposed"] = False
            process_result["decomposition_reason"] = "Simple explicit instruction"
            
        # 情况C：其他情况 (Complex, Implicit, Ambiguous) 且置信度达标 -> 分解
        else:
            final_instruction = self.decompose_instruction(instruction)
            process_result["decomposed"] = True
            process_result["decomposition_reason"] = f"Decomposed ({intent}/{difficulty})"

        return process_result, final_instruction

# Usage example
if __name__ == "__main__":
    # Initialize processor
    processor = InstructionProcessor(
        model_api_base="http://localhost:8011/v1",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        confidence_threshold=90.0
    )
    
    with processor:
        print("\n=== Test Case 1: Complex (Should Decompose) ===")
        i1 = "Monitor pedestrian activity at the crosswalk"
        res1, final1 = processor.process(i1)
        print(f"Final: {final1}")
        
        print("\n=== Test Case 2: Simple Explicit (Should Keep Raw) ===")
        i2 = "Detect trucks in the image"
        res2, final2 = processor.process(i2)
        print(f"Final: {final2}")
        
        print("\n=== Test Case 3: Low Confidence (Should Keep Raw) ===")
        # 模拟一个低置信度的情况 (虽然这里还是真实调用，但逻辑会处理低分)
        i3 = "Detect something maybe"
        res3, final3 = processor.process(i3)
        print(f"Final: {final3}")