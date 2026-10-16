# Theme annotation guide (open-models news articles)

Assign each article ONE primary theme: the theme of its MAIN SUBJECT (what the article is mainly about,
judged from the title and the body), not every theme it mentions.

| theme | main subject |
|---|---|
| model_families | a specific model or model family (any vendor: Llama, Gemma, Qwen, Nemotron, DeepSeek-V3, MiniMax, Kimi, Phi, Mistral, ...): release, versions, model card, benchmarks, capabilities, licence |
| china_geopolitics | China–US competition, export controls, AI policy / regulation / government, national AI race; DeepSeek or Chinese AI framed as a geopolitical or market shock |
| research_techniques | methods: training (RL, reasoning, fine-tuning methods, data), architecture and efficiency (attention, MoE, memory, tokens), multimodal techniques, safety research |
| developer_tools | using models: agents and agent frameworks, coding tools, APIs, SDKs, platforms, tutorials, deployment and inference how-tos |
| business_compute | hardware and infrastructure (GPUs, data centres), stocks and investment, funding and acquisitions, enterprise adoption, markets |
| none | the article fits none of the five themes (e.g. general Q&A, unrelated content, pure law text) |

Rules
- A model being mentioned does not make it model_families: "building an agent with Llama" is developer_tools.
- A Chinese model release that only describes the model is model_families; it is china_geopolitics only when the national / political / market-shock angle is central.
- Policy and regulation (also US/EU-only) is china_geopolitics.
- If two themes weigh about the same, give the stronger as primary and the other as secondary.
