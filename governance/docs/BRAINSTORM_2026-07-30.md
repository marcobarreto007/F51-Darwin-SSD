# SUPERBRAINSTORM — F51 Darwin-X

**Contexto:** 1.7B params, ROME 6/6, scanner 196K, memória 5/5. Separação de domínio não funciona neste modelo (+0.972). Soma, não separa.

---

## BLOCO 1: Hoje (1-2 horas)

1. RODEAR o ROME com 50 fatos 2023→2025 em batch. Medir knowledge preservation. Publicar tabela.
2. Baixar Qwen2.5-3B e rodar o scanner. Comparar correlação cruzada.
3. UniversalMemory com alt_keys em benchmark. 10 fatos, 5 phrasing cada.
4. Integrar observe_active no generate() do instruct. Ciclo Memory→WorldModel→Executive.
5. Publicar catálogo FFN no HuggingFace. `marco/f51-darwin-ffn-catalog`.
6. Paper de 4 páginas: "Cryptographic Circuit Cataloging and Surgical Knowledge Editing in a 1.7B Transformer".
7. Scanner no TinyLlama-1.1B-Chat. Comparar catálogos.
8. ROME editar "Quem descobriu o Brasil?" de Colón para Cabral.
9. Criar `f51 scan` CLI. Um comando.
10. Treinar SAE com 10K tokens.

## BLOCO 2: Próxima semana (10-20 horas)

11. MEMIT com matriz de covariância sobre 10K tokens.
12. Transplante cross-model com adapter linear. SmolLM2 → TinyLlama.
13. Fine-tuning seletivo: 1000 fatos editados com ROME.
14. Dashboard web local: localhost:7860.
15. SAE no INSTRUCT vs BASE.
16. Ablação de segurança: remover phishing, provar não-contaminação.
17. UniversalMemory cross-lingual: ensinar EN, perguntar PT.
18. CognitivePulse funcional: prediction_error → re-escrita na memória.
19. Exportar modelo editado como checkpoint HF.
20. Teste cego com 10 pessoas.

## BLOCO 3: Este mês (40-80 horas)

21. Curva de superposição em 5 tamanhos de modelo. Paper.
22. Destilação de API: GPT-4 → ROME → SmolLM2.
23. Treinar modelo do zero com scanner desde step 0.
24. UniversalMemory como "hippocampus": replay durante sono.
25. Transplante de circuito completo: extrair matemática de 7B, injetar no 1.7B.
26. 3 papers: catálogo, ROME benchmark, limites de superposição.
27. Competição Kaggle: "Ache o fato editado".
28. Interface AR: features flutuando em 3D.
29. Patente CIPO $50.
30. Apresentar no MILA Tea Talk.

## BLOCO 4: Sonhos (3-6 meses)

31. 4x RTX 4090. Scanner em Llama-3-70B.
32. App store de circuitos. Download de `.f51circuit`.
33. Organismo vivo: aprende, consolida, esquece.
34. Primeiro modelo 100% editado por ROME. Zero fine-tuning.
35. Nature Machine Intelligence.
36. Empresa: "Surgical AI".
37. `.f51circuit` como padrão IEEE/ISO.
38. F51 Darwin-X como sistema operacional de modelos.
39. Modelo se auto-edita.
40. Singularidade cirúrgica: modelos evoluindo sem treino.
