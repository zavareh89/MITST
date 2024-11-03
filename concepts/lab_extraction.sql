SELECT DISTINCT ON (patientunitstayid, labname, labresultoffset)
patientunitstayid,
labname,
labresultoffset :: real, labresultrevisedoffset::real
  -- extract some results from labresulttext column and perform unit conversion
  , CASE
     -- WHEN labname IN ('ALT (SGPT)', 'potassium') AND  labresult <= 0 THEN NULL
      WHEN labname = 'urinary creatinine' AND labMeasureNameInterface = 'g/24 h' THEN NULL
     -- WHEN labname IN ('sodium', 'chloride') AND  labresult <= 30 THEN NULL
     -- WHEN labname = 'magnesium' AND labresult >= 10 THEN NULL
      WHEN labname = 'calcium' AND labMeasureNameInterface = 'mmol/L' THEN (labresult*0.2496)::real -- convert to mg/dL
      WHEN labname = 'CRP' AND labMeasureNameInterface IN ('mg/L', 'MG/L') AND labresult IS NOT NULL 
        THEN (labresult*0.1)::real -- convert to mg/dL
      WHEN labname = 'CRP' AND labMeasureNameInterface IN ('mg/L', 'MG/L') AND labresult IS NULL
        AND SUBSTRING(labresulttext FROM '[0-9]+\.?[0-9]*') IS NOT NULL 
        THEN (SUBSTRING(labresulttext FROM '[0-9]+\.?[0-9]*')::real)*0.1 
      WHEN labname = '-monos' AND labMeasureNameInterface IN ('1000/uL', 'TH/uL', 'THOU/MM3') THEN NULL
      WHEN labname = '-eos' AND labMeasureNameInterface IN ('1000/uL', 'TH/uL', 'THOU/MM3') THEN NULL
      WHEN labname = '-basos' AND labMeasureNameInterface IN ('1000/uL', 'TH/uL', 'THOU/MM3', 'ug/L') THEN NULL
      WHEN labname = '-bands' AND labMeasureNameInterface = '1000/uL' THEN NULL
      WHEN labname = '-polys' AND labMeasureNameInterface = 'TH/uL' THEN NULL
      WHEN labresult IS NULL AND
        labname IN ('ALT (SGPT)', 'AST (SGOT)', 'albumin', 'anion gap', 'alkaline phos.',
                      'direct bilirubin', 'total bilirubin', 'BUN', 'creatinine', 
                      '24 h urine protein', 'urinary creatinine', 'sodium', 'potassium',
                      'chloride', 'magnesium', 'calcium', 'phosphate', 'HDL', 'LDL', 
                      'triglycerides', 'total cholesterol', 'CRP', 'ESR', 'urinary osmolality',
                      'urinary sodium', 'urinary specific gravity', 'WBC x 1000', 'RBC', 'Hgb',
                      'Hct', 'platelets x 1000', 'MCH', 'MCHC', 'RDW', 'bicarbonate', 'HCO3',
                      'Base Deficit', 'Base Excess', 'PTT', 'PT - INR', 'PT', 'pH', 'lactate',
                      'total protein', 'folate', 'LDH', 'Ferritin', 'bedside glucose', 'glucose')
        AND SUBSTRING(labresulttext FROM '[-+]?[0-9]+\.?[0-9]*') IS NOT NULL 
        THEN SUBSTRING(labresulttext FROM '[-+]?[0-9]+\.?[0-9]*')::real
   ELSE labresult::real
   END AS labresult

FROM lab 
WHERE labname in
    (
      -- Liver tests
      'ALT (SGPT)', -- U/L
      'AST (SGOT)', -- U/L
      'alkaline phos.', -- U/L
      'direct bilirubin', -- mg/dL
      'total bilirubin', -- mg/dL
      
      -- Kidney tests
      'BUN', -- mg/dL
      'creatinine', -- mg/dL
      '24 h urine protein', -- mg/dL
      'urinary creatinine', -- mg/dL

      -- Electrolytes
      'sodium', -- mEq/L
      'potassium', -- mEq/L
      'chloride', -- mEq/L
      'magnesium', -- mg/dL
      'calcium', -- mg/dL
      'phosphate', -- mg/dL

      -- Lipids
      'HDL', -- mg/dL
      'LDL', -- mg/dL
      'triglycerides', -- mg/dL
      'total cholesterol', -- mg/dL

      -- Inflammatory markers
      'CRP', -- mg/dL
      'ESR', -- mm/hr

      -- Urine tests
      'urinary osmolality', -- mOsm/kg
      'urinary sodium', -- mEq/L
      'urinary specific gravity',

      -- Complete Blood Count (CBC)
      'WBC x 1000', -- K/uL
      '-lymphs', -- %
      '-monos', -- %
      '-eos', -- %
      '-basos', -- %
      '-bands', -- %
      '-polys', -- %
      'RBC', -- M/uL
      'Hgb', -- g/dL
      'Hct', -- %
      'platelets x 1000', -- K/uL
      'MCV', -- fL
      'MCH', -- pg
      'MCHC', -- g/dL
      'RDW', -- %

      -- Other
    	'albumin', -- g/dL
      'anion gap', -- mEq/L
      'bicarbonate', -- mEq/L
      'HCO3', -- mEq/L
      'Base Deficit', -- mEq/L
      'Base Excess', -- mEq/L
      'PTT' -- sec
      'PT - INR', -- ratio
      'PT' -- sec
      'pH', -- pH Units
      'lactate', -- mmol/L
      'total protein', -- g/dL
      'folate', -- ng/mL
      'LDH', -- U/L
      'Ferritin', -- ng/mL
      'bedside glucose', -- mg/dL
      'glucose' -- mg/dL
    )
ORDER BY patientunitstayid, labname, labresultoffset, labresultrevisedoffset ASC;