# DynamicDomainAdaptation_LLMs

# 🧠 Context-Aware Prompt Steering with PHI-2

This script modifies the input embeddings of a language model (`microsoft/phi-2`) using relevant context extracted from a file, enhancing the quality and relevance of generated responses.

## 🚀 What This Script Does

1. **Loads contextual data** from a file (`context.txt`) and encodes it using `sentence-transformers`.
2. **Computes cosine similarity** between the prompt and each context line.
3. **Filters relevant context**, and averages their embeddings to create a *steering vector*.
4. **Projects the steering vector** into the embedding space of the target model (`phi-2`, 2048-dim).
5. **Applies the steering** to the first token embedding of the prompt, based on a chosen `ALPHA` value.
6. **Generates a response** using the `phi-2` model guided by this contextual vector.
7. **Logs all steps and results** in `log.txt`.

---

## 📁 File Structure

├── context.txt # Context lines used for semantic steering
├── questions.txt # Prompts to process in batch
├── script.py # Main script (this code)
└── log.txt # Process logs (auto-generated)


---

## ⚙️ Dependencies

Install with:

```bash
pip install torch transformers sentence-transformers


SIMILARITY_THRESHOLD: Minimum cosine similarity required to include a context line (default: 0.5)

ALPHA: Blending factor between the original input embedding and the steering vector

0.0 = no steering

1.0 = full steering influence

Technical Notes

Uses sentence-transformers (all-mpnet-base-v2) to embed both prompt and context (768-dim).

The 768-dim context embedding is projected to 2048-dim via a torch.nn.Linear layer to match the phi-2 input.

Only the first token of the input embedding is modified to steer generation subtly but effectively.

Logs include detailed info on embedding shapes, similarity scores, and generated outputs.

📈 Potential Use Cases

Steering output behavior toward specific topics or styles without fine-tuning.

Domain-aware generation based on external knowledge or guidelines.

Dynamically biasing LLMs using textual memory or persona inputs.
