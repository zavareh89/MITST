ALTER TABLE
    patient
ADD
    COLUMN previous_visit INT REFERENCES patient(patientunitstayid),
ADD
    COLUMN next_visit INT REFERENCES patient(patientunitstayid);

-- update previous visit values based on unitvisitorder
WITH patient_order_fixed AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY uniquepid,
            patienthealthsystemstayid,
            hospitalAdmitTime24,
            hospitalDischargeTime24
            ORDER BY
                unitvisitnumber
        ) AS unitvisitorder
    FROM
        patient
)
UPDATE
    patient
SET
    previous_visit = pf2.patientunitstayid
FROM
    patient_order_fixed pf1,
    patient_order_fixed pf2
WHERE
    patient.patientunitstayid = pf1.patientunitstayid
    AND pf1.unitvisitorder > 1
    AND pf1.uniquepid = pf2.uniquepid
    AND pf1.patienthealthsystemstayid = pf2.patienthealthsystemstayid
    AND pf1.hospitalAdmitTime24 = pf2.hospitalAdmitTime24
    AND pf2.hospitalDischargeTime24 = pf2.hospitalDischargeTime24
    AND pf1.unitvisitorder = pf2.unitvisitorder + 1;

-- update next visit values based on unitvisitorder
WITH patient_order_fixed AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY uniquepid,
            patienthealthsystemstayid,
            hospitalAdmitTime24,
            hospitalDischargeTime24
            ORDER BY
                unitvisitnumber
        ) AS unitvisitorder
    FROM
        patient
)
UPDATE
    patient
SET
    next_visit = pf2.patientunitstayid
FROM
    patient_order_fixed pf1,
    patient_order_fixed pf2
WHERE
    patient.patientunitstayid = pf1.patientunitstayid
    AND pf1.uniquepid = pf2.uniquepid
    AND pf1.patienthealthsystemstayid = pf2.patienthealthsystemstayid
    AND pf1.hospitalAdmitTime24 = pf2.hospitalAdmitTime24
    AND pf2.hospitalDischargeTime24 = pf2.hospitalDischargeTime24
    AND pf1.unitvisitorder = pf2.unitvisitorder - 1;