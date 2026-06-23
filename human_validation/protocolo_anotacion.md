# Protocolo de anotación humana — 400 titulares

**Manuscrito:** Access-2026-11808 (R2)
**Diseño:** Opción C combinada
- 100 neutrals del fichero `auditoria_completa_llama3_5000_final.csv` (n=5000)
- 300 emocionales estratificados del fichero `llama3_full_validation_results.csv` (n=2000)

**Distribución objetivo (RoBERTa-BNE como categoría base del muestreo):**
- neutral: 100
- anger: 70 | sadness: 70 | joy: 60 | fear: 50 | surprise: 30 | disgust: 20

**Procedimiento del anotador:**
1. Abrir `Anotacion_Humana_400.xlsx`, leer la hoja 'Instrucciones'.
2. En la hoja 'Titulares', columna F (`Emoción humana`), seleccionar UNA de las 7 etiquetas del dropdown para cada titular.
3. No buscar contexto externo: anotar solo lo que el titular transmite por sí solo.
4. Forzar elección (no dejar vacío). Si ninguna emoción aplica, marcar `neutral`.
5. Opcional: usar la columna E para una nota breve cuando la decisión sea dudosa.
6. Guardar el archivo con el mismo nombre y enviarlo a Claude.

**Métricas que se calcularán después:**
- Cohen's kappa (humano vs RoBERTa-BNE) — global y por clase
- F1 macro y per-class (humano vs RoBERTa-BNE)
- Matriz de confusión 7x7
- Cohen's kappa (humano vs Llama 3) — donde haya etiqueta Llama disponible
- Krippendorff's alpha (humano, RoBERTa, Llama) — concordancia 3-vías

**Seed reproducible:** 42 (sampling + shuffle)
**Total titulares:** 400
**Tiempo estimado:** 2-3 horas
