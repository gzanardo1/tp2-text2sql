# Análise de Erros — Casos de Regressão (Fase 4)

Exemplos onde o **baseline acertou** mas o **fine-tuned (config A) errou**. São os casos mais informativos: revelam erros INTRODUZIDOS pelo treinamento (não erros pré-existentes do modelo-base).

## Exemplo 1 — banco `pets_1`

**Pergunta**: Find the weight of the youngest dog.

**Gold SQL**:
```sql
SELECT weight FROM pets ORDER BY pet_age LIMIT 1
```

**SQL gerado (qlora_a)**:
```sql
SELECT t2.weight FROM pets AS t2 JOIN has_pet AS t1 ON t2.petid  =  t1.petid WHERE t1.stuid IN (SELECT stuid FROM student WHERE pettype  =  'dog' ORDER BY age LIMIT 1)
```

**Motivo da falha**: Resultados divergentes (ordered=True). gold=1 linhas, pred=0 linhas.

---

## Exemplo 2 — banco `car_1`

**Pergunta**: How many countries does each continent have? List the continent id, continent name and the number of countries.

**Gold SQL**:
```sql
SELECT T1.ContId ,  T1.Continent ,  count(*) FROM CONTINENTS AS T1 JOIN COUNTRIES AS T2 ON T1.ContId  =  T2.Continent GROUP BY T1.ContId;
```

**SQL gerado (qlora_a)**:
```sql
SELECT T2.ContId ,  T1.Continent ,  COUNT(*) FROM continents AS T1 JOIN countries AS T2 ON T1.ContId  =  T2.Continent GROUP BY T1.ContId
```

**Motivo da falha**: Falha ao executar a consulta gerada: no such column: T2.ContId

---

## Exemplo 3 — banco `flight_2`

**Pergunta**: Which abbreviation corresponds to Jetblue Airways?

**Gold SQL**:
```sql
SELECT Abbreviation FROM AIRLINES WHERE Airline  =  "JetBlue Airways"
```

**SQL gerado (qlora_a)**:
```sql
SELECT Abbreviation FROM airlines WHERE Airline  =  "Jetblue Airways"
```

**Motivo da falha**: Resultados divergentes (ordered=False). gold=1 linhas, pred=0 linhas.

---
