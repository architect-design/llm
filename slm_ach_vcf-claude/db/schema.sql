-- =============================================================================
-- FinSLM Oracle Schema — ACH NACHA Data Model
-- Run as: sqlplus finslm/password@ORCL @schema.sql
--
-- Tables:
--   ACH_ODFI_CONFIG      ODFI (originating bank) master config
--   ACH_COMPANIES        Company / originator master data
--   ACH_ACCOUNTS         Receiver account master (RDFI side)
--   ACH_TRANSACTIONS     Pending ACH transactions
--   ACH_BATCHES          Batch configuration overrides
--   ACH_FILE_LOG         Generated file audit log
--   ACH_VALIDATION_LOG   Validation run results
--   ACH_TRAINING_CORPUS  SLM training examples derived from Oracle data
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. ODFI Configuration (Originating Depository Financial Institution)
-- ---------------------------------------------------------------------------
CREATE TABLE FINSLM.ACH_ODFI_CONFIG (
    ODFI_ID              NUMBER(10)     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ROUTING_NUMBER       CHAR(9)        NOT NULL,          -- 9-digit ABA routing
    BANK_NAME            VARCHAR2(23)   NOT NULL,
    IMMEDIATE_DEST       CHAR(10)       NOT NULL,          -- space + 9 digits
    IMMEDIATE_ORIGIN     CHAR(10)       NOT NULL,
    DEST_SHORT_NAME      VARCHAR2(23)   NOT NULL,
    ORIGIN_SHORT_NAME    VARCHAR2(23)   NOT NULL,
    IS_ACTIVE            CHAR(1)        DEFAULT 'Y' NOT NULL,
    CREATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    UPDATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT CHK_ODFI_ROUTING CHECK (REGEXP_LIKE(ROUTING_NUMBER, '^\d{9}$')),
    CONSTRAINT CHK_ODFI_ACTIVE  CHECK (IS_ACTIVE IN ('Y','N'))
);

COMMENT ON TABLE  FINSLM.ACH_ODFI_CONFIG              IS 'Originating bank configuration for ACH file headers';
COMMENT ON COLUMN FINSLM.ACH_ODFI_CONFIG.ROUTING_NUMBER IS '9-digit ABA routing number (must pass check-digit)';
COMMENT ON COLUMN FINSLM.ACH_ODFI_CONFIG.IMMEDIATE_DEST IS 'Space + 9-digit routing for File Header field pos 4-13';

-- Seed one ODFI row so the system works out of the box
INSERT INTO FINSLM.ACH_ODFI_CONFIG
  (ROUTING_NUMBER, BANK_NAME, IMMEDIATE_DEST, IMMEDIATE_ORIGIN,
   DEST_SHORT_NAME, ORIGIN_SHORT_NAME)
VALUES
  ('021000021','JPMORGAN CHASE BANK NA',' 021000021','021000021         ',
   'JPMORGAN CHASE NA  ','JPMORGAN CHASE NA  ');


-- ---------------------------------------------------------------------------
-- 2. Company / Originator Master
-- ---------------------------------------------------------------------------
CREATE TABLE FINSLM.ACH_COMPANIES (
    COMPANY_ID           NUMBER(10)     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    COMPANY_NAME         VARCHAR2(16)   NOT NULL,          -- NACHA 16-char field
    COMPANY_ID_NUMBER    VARCHAR2(10)   NOT NULL,          -- 10-char originator ID
    COMPANY_ENTRY_DESC   VARCHAR2(10)   DEFAULT 'PAYMENT'  NOT NULL,
    SEC_CODE             CHAR(3)        DEFAULT 'PPD'      NOT NULL,
    SERVICE_CLASS_CODE   CHAR(3)        DEFAULT '200'      NOT NULL,
    DEFAULT_ODFI_ID      NUMBER(10)     REFERENCES FINSLM.ACH_ODFI_CONFIG(ODFI_ID),
    DISCRETIONARY_DATA   VARCHAR2(20),
    IS_ACTIVE            CHAR(1)        DEFAULT 'Y' NOT NULL,
    CREATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    UPDATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT CHK_CO_SEC     CHECK (SEC_CODE IN ('PPD','CCD','CTX','WEB','TEL',
                                                   'COR','NOC','RCK','ARC','BOC',
                                                   'POP','XCK','IAT','ENR','MTE','SHR')),
    CONSTRAINT CHK_CO_SCC     CHECK (SERVICE_CLASS_CODE IN ('200','220','225')),
    CONSTRAINT CHK_CO_ACTIVE  CHECK (IS_ACTIVE IN ('Y','N'))
);

COMMENT ON TABLE  FINSLM.ACH_COMPANIES IS 'Originating company master — maps to Batch Header record (type 5)';
COMMENT ON COLUMN FINSLM.ACH_COMPANIES.COMPANY_NAME IS 'Max 16 chars — truncated in Batch Header pos 5-20';
COMMENT ON COLUMN FINSLM.ACH_COMPANIES.SEC_CODE     IS 'Standard Entry Class code written to Batch Header pos 51-53';

CREATE INDEX FINSLM.IDX_CO_SEC  ON FINSLM.ACH_COMPANIES(SEC_CODE);
CREATE INDEX FINSLM.IDX_CO_ODFI ON FINSLM.ACH_COMPANIES(DEFAULT_ODFI_ID);


-- ---------------------------------------------------------------------------
-- 3. Receiver Account Master  (RDFI / beneficiary side)
-- ---------------------------------------------------------------------------
CREATE TABLE FINSLM.ACH_ACCOUNTS (
    ACCOUNT_ID           NUMBER(15)     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    COMPANY_ID           NUMBER(10)     NOT NULL REFERENCES FINSLM.ACH_COMPANIES(COMPANY_ID),
    -- Receiver identity
    INDIVIDUAL_NAME      VARCHAR2(22)   NOT NULL,          -- NACHA pos 55-76
    INDIVIDUAL_ID        VARCHAR2(15),                     -- NACHA pos 40-54
    -- RDFI routing & account
    RDFI_ROUTING         CHAR(8)        NOT NULL,          -- 8 digits, no check digit
    RDFI_CHECK_DIGIT     CHAR(1)        NOT NULL,
    ACCOUNT_NUMBER       VARCHAR2(17)   NOT NULL,          -- NACHA pos 13-29
    ACCOUNT_TYPE         CHAR(1)        DEFAULT 'C' NOT NULL, -- C=Checking S=Savings
    -- Status
    PRENOTE_STATUS       VARCHAR2(10)   DEFAULT 'LIVE' NOT NULL,
    IS_ACTIVE            CHAR(1)        DEFAULT 'Y'    NOT NULL,
    CREATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    UPDATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT CHK_ACC_TYPE   CHECK (ACCOUNT_TYPE IN ('C','S','G','L')),
    CONSTRAINT CHK_ACC_PRENOTE CHECK (PRENOTE_STATUS IN ('LIVE','PRENOTE','RETURN','FROZEN')),
    CONSTRAINT CHK_ACC_ACTIVE  CHECK (IS_ACTIVE IN ('Y','N'))
);

COMMENT ON TABLE  FINSLM.ACH_ACCOUNTS IS 'Receiver (RDFI) account master — drives Entry Detail records (type 6)';
COMMENT ON COLUMN FINSLM.ACH_ACCOUNTS.RDFI_ROUTING      IS '8-digit RDFI routing (check digit stored separately)';
COMMENT ON COLUMN FINSLM.ACH_ACCOUNTS.RDFI_CHECK_DIGIT  IS 'ABA check digit — validated via 3-7-1 algorithm';
COMMENT ON COLUMN FINSLM.ACH_ACCOUNTS.PRENOTE_STATUS    IS 'LIVE=real money, PRENOTE=zero-dollar test, RETURN=returned';

CREATE INDEX FINSLM.IDX_ACC_CO     ON FINSLM.ACH_ACCOUNTS(COMPANY_ID);
CREATE INDEX FINSLM.IDX_ACC_RDFI   ON FINSLM.ACH_ACCOUNTS(RDFI_ROUTING);
CREATE INDEX FINSLM.IDX_ACC_ACTIVE ON FINSLM.ACH_ACCOUNTS(IS_ACTIVE, PRENOTE_STATUS);


-- ---------------------------------------------------------------------------
-- 4. Pending ACH Transactions
-- ---------------------------------------------------------------------------
CREATE TABLE FINSLM.ACH_TRANSACTIONS (
    TRANSACTION_ID       NUMBER(18)     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ACCOUNT_ID           NUMBER(15)     NOT NULL REFERENCES FINSLM.ACH_ACCOUNTS(ACCOUNT_ID),
    COMPANY_ID           NUMBER(10)     NOT NULL REFERENCES FINSLM.ACH_COMPANIES(COMPANY_ID),
    -- Financial fields
    TRANSACTION_CODE     CHAR(2)        NOT NULL,          -- 22/27/32/37/etc.
    AMOUNT_CENTS         NUMBER(12,0)   NOT NULL,          -- stored as integer cents
    EFFECTIVE_DATE       DATE           NOT NULL,          -- YYMMDD in NACHA
    -- Descriptive
    INDIVIDUAL_ID_OVERRIDE VARCHAR2(15),                   -- overrides account-level ID
    DISCRETIONARY_DATA   CHAR(2)        DEFAULT '  ',
    ADDENDA_INFO         VARCHAR2(80),                     -- optional addenda text
    -- Lifecycle
    STATUS               VARCHAR2(12)   DEFAULT 'PENDING' NOT NULL,
    BATCH_NUMBER         NUMBER(7),                        -- populated when batched
    FILE_ID              NUMBER(15)     REFERENCES FINSLM.ACH_FILE_LOG(FILE_ID),
    RETURN_CODE          CHAR(3),
    CREATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    PROCESSED_AT         TIMESTAMP,
    CONSTRAINT CHK_TXN_CODE   CHECK (TRANSACTION_CODE IN (
                                  '22','23','24','27','28','29',   -- Checking
                                  '32','33','34','37','38','39',   -- Savings
                                  '42','43','44','47','48','49',   -- GL
                                  '52','53','54','55'              -- Loan
                              )),
    CONSTRAINT CHK_TXN_AMOUNT CHECK (AMOUNT_CENTS >= 0),
    CONSTRAINT CHK_TXN_STATUS CHECK (STATUS IN ('PENDING','BATCHED','SENT',
                                                 'RETURNED','SETTLED','VOID'))
);

COMMENT ON TABLE  FINSLM.ACH_TRANSACTIONS IS 'Pending transactions queued for ACH file generation';
COMMENT ON COLUMN FINSLM.ACH_TRANSACTIONS.AMOUNT_CENTS IS 'Amount in cents — divided by 100 for NACHA 10-digit implied-decimal field';
COMMENT ON COLUMN FINSLM.ACH_TRANSACTIONS.STATUS       IS 'Lifecycle: PENDING→BATCHED→SENT→SETTLED or RETURNED';

CREATE INDEX FINSLM.IDX_TXN_ACCT    ON FINSLM.ACH_TRANSACTIONS(ACCOUNT_ID);
CREATE INDEX FINSLM.IDX_TXN_CO      ON FINSLM.ACH_TRANSACTIONS(COMPANY_ID);
CREATE INDEX FINSLM.IDX_TXN_STATUS  ON FINSLM.ACH_TRANSACTIONS(STATUS, EFFECTIVE_DATE);
CREATE INDEX FINSLM.IDX_TXN_EFFDATE ON FINSLM.ACH_TRANSACTIONS(EFFECTIVE_DATE, STATUS);
CREATE INDEX FINSLM.IDX_TXN_FILE    ON FINSLM.ACH_TRANSACTIONS(FILE_ID);


-- ---------------------------------------------------------------------------
-- 5. File Log (audit trail for every generated ACH file)
-- ---------------------------------------------------------------------------
CREATE TABLE FINSLM.ACH_FILE_LOG (
    FILE_ID              NUMBER(15)     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    FILE_NAME            VARCHAR2(100)  NOT NULL,
    FILE_ID_MODIFIER     CHAR(1)        NOT NULL,          -- A-Z cycling
    ODFI_ID              NUMBER(10)     REFERENCES FINSLM.ACH_ODFI_CONFIG(ODFI_ID),
    -- File-level counters (duplicated from File Control for fast reporting)
    BATCH_COUNT          NUMBER(6)      DEFAULT 0,
    ENTRY_COUNT          NUMBER(8)      DEFAULT 0,
    BLOCK_COUNT          NUMBER(6)      DEFAULT 0,
    TOTAL_DEBIT_CENTS    NUMBER(15,0)   DEFAULT 0,
    TOTAL_CREDIT_CENTS   NUMBER(15,0)   DEFAULT 0,
    -- Metadata
    GENERATION_METHOD    VARCHAR2(20)   DEFAULT 'ORACLE',  -- ORACLE / SYNTHETIC / SLM
    IS_VALID             CHAR(1)        DEFAULT 'Y',
    VALIDATION_ERRORS    NUMBER(5)      DEFAULT 0,
    FILE_CONTENT         CLOB,                             -- full file text stored for audit
    CREATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    SENT_AT              TIMESTAMP,
    CONSTRAINT CHK_LOG_VALID  CHECK (IS_VALID IN ('Y','N')),
    CONSTRAINT CHK_LOG_METHOD CHECK (GENERATION_METHOD IN ('ORACLE','SYNTHETIC','SLM','HYBRID'))
);

COMMENT ON TABLE  FINSLM.ACH_FILE_LOG IS 'Audit log for every generated ACH NACHA file';
COMMENT ON COLUMN FINSLM.ACH_FILE_LOG.FILE_CONTENT IS 'Full 94-char-per-record NACHA file stored as CLOB for replay';

CREATE INDEX FINSLM.IDX_FLOG_DATE   ON FINSLM.ACH_FILE_LOG(CREATED_AT);
CREATE INDEX FINSLM.IDX_FLOG_METHOD ON FINSLM.ACH_FILE_LOG(GENERATION_METHOD);


-- ---------------------------------------------------------------------------
-- 6. Validation Log
-- ---------------------------------------------------------------------------
CREATE TABLE FINSLM.ACH_VALIDATION_LOG (
    VALIDATION_ID        NUMBER(15)     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    FILE_ID              NUMBER(15)     REFERENCES FINSLM.ACH_FILE_LOG(FILE_ID),
    FILE_NAME            VARCHAR2(200),
    FILE_TYPE            CHAR(3)        DEFAULT 'ACH',
    IS_VALID             CHAR(1)        NOT NULL,
    ERROR_COUNT          NUMBER(5)      DEFAULT 0,
    WARNING_COUNT        NUMBER(5)      DEFAULT 0,
    VALIDATION_REPORT    CLOB,                             -- full JSON report
    CREATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT CHK_VLOG_VALID    CHECK (IS_VALID IN ('Y','N')),
    CONSTRAINT CHK_VLOG_TYPE     CHECK (FILE_TYPE IN ('ACH','VCF'))
);

CREATE INDEX FINSLM.IDX_VLOG_FILE ON FINSLM.ACH_VALIDATION_LOG(FILE_ID);
CREATE INDEX FINSLM.IDX_VLOG_DATE ON FINSLM.ACH_VALIDATION_LOG(CREATED_AT);


-- ---------------------------------------------------------------------------
-- 7. SLM Training Corpus  (ACH files extracted from Oracle for model training)
-- ---------------------------------------------------------------------------
CREATE TABLE FINSLM.ACH_TRAINING_CORPUS (
    CORPUS_ID            NUMBER(15)     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    FILE_LOG_ID          NUMBER(15)     REFERENCES FINSLM.ACH_FILE_LOG(FILE_ID),
    FILE_CONTENT         CLOB           NOT NULL,
    SEC_CODE             CHAR(3),
    SERVICE_CLASS_CODE   CHAR(3),
    BATCH_COUNT          NUMBER(5),
    ENTRY_COUNT          NUMBER(8),
    IS_USED_FOR_TRAINING CHAR(1)        DEFAULT 'N',
    SPLIT_TYPE           VARCHAR2(10)   DEFAULT 'TRAIN',   -- TRAIN / VAL / TEST
    CREATED_AT           TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
    CONSTRAINT CHK_CORP_SPLIT  CHECK (SPLIT_TYPE IN ('TRAIN','VAL','TEST')),
    CONSTRAINT CHK_CORP_USED   CHECK (IS_USED_FOR_TRAINING IN ('Y','N'))
);

CREATE INDEX FINSLM.IDX_CORPUS_SPLIT ON FINSLM.ACH_TRAINING_CORPUS(SPLIT_TYPE, IS_USED_FOR_TRAINING);


-- ---------------------------------------------------------------------------
-- Helper view: full transaction detail for ACH generation
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW FINSLM.V_ACH_PENDING_TRANSACTIONS AS
SELECT
    t.TRANSACTION_ID,
    t.COMPANY_ID,
    t.ACCOUNT_ID,
    t.TRANSACTION_CODE,
    t.AMOUNT_CENTS,
    TO_CHAR(t.EFFECTIVE_DATE, 'YYMMDD')          AS EFFECTIVE_DATE_STR,
    NVL(t.INDIVIDUAL_ID_OVERRIDE, a.INDIVIDUAL_ID) AS INDIVIDUAL_ID,
    a.INDIVIDUAL_NAME,
    a.RDFI_ROUTING,
    a.RDFI_CHECK_DIGIT,
    a.ACCOUNT_NUMBER,
    a.ACCOUNT_TYPE,
    NVL(t.DISCRETIONARY_DATA, '  ')              AS DISCRETIONARY_DATA,
    t.ADDENDA_INFO,
    -- Company
    c.COMPANY_NAME,
    c.COMPANY_ID_NUMBER,
    c.COMPANY_ENTRY_DESC,
    c.SEC_CODE,
    c.SERVICE_CLASS_CODE,
    c.DISCRETIONARY_DATA                          AS COMPANY_DISC_DATA,
    -- ODFI
    o.ROUTING_NUMBER                              AS ODFI_ROUTING,
    o.IMMEDIATE_DEST,
    o.IMMEDIATE_ORIGIN,
    o.DEST_SHORT_NAME,
    o.ORIGIN_SHORT_NAME,
    t.STATUS
FROM FINSLM.ACH_TRANSACTIONS   t
JOIN FINSLM.ACH_ACCOUNTS        a ON t.ACCOUNT_ID   = a.ACCOUNT_ID
JOIN FINSLM.ACH_COMPANIES       c ON t.COMPANY_ID   = c.COMPANY_ID
LEFT JOIN FINSLM.ACH_ODFI_CONFIG o ON c.DEFAULT_ODFI_ID = o.ODFI_ID
WHERE t.STATUS = 'PENDING'
  AND a.IS_ACTIVE = 'Y'
  AND c.IS_ACTIVE = 'Y';

COMMENT ON VIEW FINSLM.V_ACH_PENDING_TRANSACTIONS IS
  'Denormalised view used by OracleACHGenerator to build NACHA Entry Detail records';

-- ---------------------------------------------------------------------------
-- Stored procedure: mark transactions as BATCHED after file generation
-- ---------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE FINSLM.SP_MARK_BATCHED (
    p_transaction_ids  IN  SYS.ODCINUMBERLIST,
    p_file_id          IN  NUMBER,
    p_batch_number     IN  NUMBER
) AS
BEGIN
    FORALL i IN 1 .. p_transaction_ids.COUNT
        UPDATE FINSLM.ACH_TRANSACTIONS
        SET    STATUS       = 'BATCHED',
               FILE_ID      = p_file_id,
               BATCH_NUMBER = p_batch_number,
               PROCESSED_AT = SYSTIMESTAMP
        WHERE  TRANSACTION_ID = p_transaction_ids(i);
    COMMIT;
END;
/

COMMIT;
