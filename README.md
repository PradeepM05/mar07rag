DECLARE
    v_date DATE := TO_DATE('01-APR-2025', 'DD-MON-YYYY');
BEGIN
    WHILE v_date <= TO_DATE('20-APR-2025', 'DD-MON-YYYY') LOOP
        -- Skip weekends
        IF TO_CHAR(v_date, 'DY', 'NLS_DATE_LANGUAGE=ENGLISH') NOT IN ('SAT', 'SUN') THEN
            -- Call for FI_PRICING_SCR_HIST
            CDW_ACTIMIZE.SP_SPLIT_TBL_PARTITION(
                'cdw_actimize',
                'FI_PRICING_SCR_HIST',
                v_date,
                'N'
            );

            -- Call for EQ_PRICING_SCR_HIST
            CDW_ACTIMIZE.SP_SPLIT_TBL_PARTITION(
                'cdw_actimize',
                'EQ_PRICING_SCR_HIST',
                v_date,
                'N'
            );
        END IF;

        v_date := v_date + 1;
    END LOOP;
END;
/
