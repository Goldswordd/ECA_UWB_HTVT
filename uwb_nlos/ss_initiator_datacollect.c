// ============================================================================
// DATA COLLECTION FIRMWARE — ECA-UWB / eWINE-compatible format
// Output CSV columns (match eWINE cols 0–9, then CIR at col 10):
//
//   Col 0       : NLOS label  (0=LOS, 1=NLOS)  ← set g_label before flashing
//   Col 1       : RANGE       (mm, float)
//   Col 2       : FP_IDX      (first path index in accumulator)
//   Col 3       : FP_AMP1
//   Col 4       : FP_AMP2
//   Col 5       : FP_AMP3
//   Col 6       : STDEV_NOISE
//   Col 7       : CIR_PWR     (raw register 0x12 value, NOT dBm)
//   Col 8       : MAX_NOISE
//   Col 9       : RXPACC      (used for CIR normalization in Python)
//   Col 10–109  : CIR magnitude [FP_IDX-5 … FP_IDX+94]  (100 samples)
//
// Python config: set VNU_CIR_START = 10  (eWINE uses 15, differs only here)
// All diagnostic column indices (1–9) are identical to eWINE → zero code change
// in train_ecauwb.py / ECA-UWB model when loading VNU data.
// ============================================================================

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "FreeRTOS.h"
#include "task.h"
#include "deca_device_api.h"
#include "deca_regs.h"
#include "port_platform.h"

#define APP_NAME "SS TWR INIT — DATA COLLECTION"

// ──────────────────────────────────────────────────────────────────────────────
// ★ SET LABEL BEFORE EACH COLLECTION SESSION:
//     0 = LOS environment (clear line of sight)
//     1 = NLOS environment (obstruction between tag and anchor)
// ──────────────────────────────────────────────────────────────────────────────
static uint8_t g_label = 0;

// ──────────────────────────────────────────────────────────────────────────────
// Timing & device config
// ──────────────────────────────────────────────────────────────────────────────
#define TIME_SLOT_MS          2
#define RESPONSE_TIMEOUT_MS   5
#define CYCLE_DELAY_MS        2
// #define MAX_RETRIES_PER_CYCLE 5   // unused — kept for reference

#define MY_INITIATOR_DEVICE_ID 0x5678
#define ANCHOR_1 0x1001
#define ANCHOR_2 0x1002
#define ANCHOR_3 0x1003
#define ANCHOR_4 0x1004
#define NUM_ANCHORS 1

// ──────────────────────────────────────────────────────────────────────────────
// CIR window: FP_IDX - 5  …  FP_IDX + 94  = 100 samples
// Enough context around the first path; ECA-UWB uses 50 (sub-window in Python)
// ──────────────────────────────────────────────────────────────────────────────
#define CIR_SAMPLES_PER_ANCHOR 100
#define CIR_PRE_FP             5      // samples before FP_IDX
#define CHUNK_SIZE             16     // read 16 samples per SPI burst (64 bytes)

// ──────────────────────────────────────────────────────────────────────────────
// Data structures
// ──────────────────────────────────────────────────────────────────────────────
typedef struct {
    int16_t real;
    int16_t img;
} cir_sample_t;

// ── NOT USED by ECA-UWB — kept commented for future experiments ──────────────
// typedef struct {
//     float kurtosis;    // excess kurtosis: LOS high, NLOS low
//     float skewness;    // distribution skew of CIR magnitude
//     int   peak_count;  // local peaks above mean + 0.5*std
// } cir_multipath_t;

typedef struct {
    dwt_rxdiag_t  diagnostics;
    cir_sample_t  cir_samples[CIR_SAMPLES_PER_ANCHOR];
    uint16_t      fp_idx;        // firstPath >> 6  (sample index)
    uint16_t      cir_pwr_raw;   // raw CIR_PWR register — must read before RX reset
    uint8_t       valid;
    double        distance;
    // cir_multipath_t multipath;  // uncomment if kurtosis/skewness needed
} anchor_data_t;

static anchor_data_t anchor_data[NUM_ANCHORS];
static uint32        anchor_ids[NUM_ANCHORS] = {ANCHOR_1, ANCHOR_2, ANCHOR_3, ANCHOR_4};

// ──────────────────────────────────────────────────────────────────────────────
// Message frames
// ──────────────────────────────────────────────────────────────────────────────
static uint8 tx_poll_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'I', 'O', 'V', 'E', 0xE0, 0, 0, 0, 0, 0, 0};
static uint8 rx_resp_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'V', 'E', 'I', 'O', 0xE1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#define ALL_MSG_COMMON_LEN       10
#define ALL_MSG_SN_IDX            2
#define POLL_MSG_DEVICE_ID_IDX   10
#define POLL_MSG_DEVICE_ID_LEN    4
#define RESP_MSG_POLL_RX_TS_IDX  10
#define RESP_MSG_RESP_TX_TS_IDX  14
#define RESP_MSG_TS_LEN           4
#define RESP_MSG_DEVICE_ID_IDX   18
#define RESP_MSG_DEVICE_ID_LEN    4

static uint8 frame_seq_nb = 0;

#define RX_BUF_LEN 24
static uint8  rx_buffer[RX_BUF_LEN];
static uint32 status_reg = 0;

#define UUS_TO_DWT_TIME 65536
#define SPEED_OF_LIGHT  299702547

// ──────────────────────────────────────────────────────────────────────────────
// Performance counters (diagnostics only, not part of CSV)
// ──────────────────────────────────────────────────────────────────────────────
static volatile int tx_count          = 0;
static volatile int rx_count          = 0;
static volatile int timeout_count     = 0;
static volatile int skip_count        = 0;
static volatile int complete_cycles   = 0;
// static volatile int incomplete_cycles = 0;  // unused
// static volatile int measurement_cycle = 0;  // unused

static uint8_t ml_header_printed = 0;

// ============================================================================
// HELPER: integer log10 approximation (avoids libm)
// ============================================================================
static float approx_log10(uint32_t x)
{
    uint8_t log2 = 0;
    while (x >>= 1) ++log2;
    return log2 * 0.30103f;
}

// ============================================================================
// HELPER: integer square root
// ============================================================================
static uint32_t sqrt_uint32(uint32_t x)
{
    uint32_t res = 0, bit = 1UL << 30;
    while (bit > x)        bit >>= 2;
    while (bit != 0) {
        if (x >= res + bit) { x -= res + bit; res = (res >> 1) + bit; }
        else                  res >>= 1;
        bit >>= 2;
    }
    return res;
}

// ── float sqrt (Newton) — needed only for multipath features ─────────────────
// static float sqrt_float(float x)
// {
//     if (x <= 0.0f) return 0.0f;
//     float s = x * 0.5f;
//     for (int i = 0; i < 16; i++) s = (s + x / s) * 0.5f;
//     return s;
// }

// ============================================================================
// POWER: first path power (dBm) — stored but NOT written to CSV
// (eWINE col 7 is raw CIR_PWR register, not fPathPWR dBm)
// Keep for debugging / signal quality monitoring via printf if needed.
// ============================================================================
// static int8_t fPathPWR(dwt_rxdiag_t diag)
// {
//     uint32_t sum = diag.firstPathAmp1 * diag.firstPathAmp1 +
//                    diag.firstPathAmp2 * diag.firstPathAmp2 +
//                    diag.firstPathAmp3 * diag.firstPathAmp3;
//     float log_sum = approx_log10(sum);
//     float log_N2  = approx_log10(diag.rxPreamCount) * 2.0f;
//     return (int8_t)(10.0f * (log_sum - log_N2) - 121.74f + 0.5f);
// }

// ============================================================================
// POWER: raw CIR_PWR register value (col 7 in eWINE, used by ECA-UWB)
// Must be called BEFORE dwt_rxreset() — register is cleared on reset.
// ============================================================================
static uint16_t readCIR_PWR_raw(void)
{
    uint32_t fqual;
    dwt_readfromdevice(0x12, 4, 4, (uint8_t *)&fqual);
    return (uint16_t)((fqual >> 16) & 0xFFFF);
}

// ── Total RX power (dBm) — NOT written to CSV, eWINE does not include it ────
// static int8_t RX_PWR(dwt_rxdiag_t diag)
// {
//     uint32_t C = readCIR_PWR_raw();
//     uint32_t N = diag.rxPreamCount;
//     if (C == 0 || N == 0) return 0;
//     float logC  = approx_log10(C) + (17.0f * 0.30103f);
//     float logN2 = approx_log10(N) * 2.0f;
//     return (int8_t)(10.0f * (logC - logN2) - 121.74f + 0.5f);
// }

// ============================================================================
// CIR READ: FP_IDX - CIR_PRE_FP  →  FP_IDX - CIR_PRE_FP + CIR_SAMPLES_PER_ANCHOR
// Reads in CHUNK_SIZE bursts to stay within SPI buffer limits.
// ============================================================================
static void read_cir_samples(cir_sample_t *samples, int fp_index)
{
    int start_idx = fp_index - CIR_PRE_FP;
    if (start_idx < 0) start_idx = 0;

    int max_start = 1016 - CIR_SAMPLES_PER_ANCHOR;
    if (start_idx > max_start) start_idx = max_start;

    uint8_t buffer[CHUNK_SIZE * 4 + 1];
    int samples_read = 0;

    while (samples_read < CIR_SAMPLES_PER_ANCHOR)
    {
        int remaining  = CIR_SAMPLES_PER_ANCHOR - samples_read;
        int this_chunk = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;
        int byte_offset = (start_idx + samples_read) * 4;

        dwt_readaccdata(buffer, this_chunk * 4 + 1, byte_offset);

        for (int i = 0; i < this_chunk; i++)
        {
            int off = i * 4 + 1;  // +1: skip DW1000 dummy byte
            samples[samples_read + i].real = (int16_t)(buffer[off]     | (buffer[off + 1] << 8));
            samples[samples_read + i].img  = (int16_t)(buffer[off + 2] | (buffer[off + 3] << 8));
        }
        samples_read += this_chunk;
    }
}

// ── MULTIPATH FEATURES (kurtosis / skewness / peak_count) ───────────────────
// NOT used by ECA-UWB / eWINE-compatible models.
// Uncomment if you want to experiment with them as extra features.
// ─────────────────────────────────────────────────────────────────────────────
// static cir_multipath_t compute_multipath_features(cir_sample_t *samples, int n)
// {
//     cir_multipath_t mp = {0.0f, 0.0f, 0};
//     float mag[CIR_SAMPLES_PER_ANCHOR];
//     float mean = 0.0f;
//     for (int i = 0; i < n; i++) {
//         int32_t r = samples[i].real, im = samples[i].img;
//         mag[i] = (float)sqrt_uint32((uint32_t)(r*r + im*im));
//         mean += mag[i];
//     }
//     mean /= (float)n;
//     float m2=0,m3=0,m4=0;
//     for (int i = 0; i < n; i++) {
//         float d=mag[i]-mean, d2=d*d;
//         m2+=d2; m3+=d2*d; m4+=d2*d2;
//     }
//     m2/=n; m3/=n; m4/=n;
//     float std = sqrt_float(m2);
//     mp.kurtosis = (m2>0) ? (m4/(m2*m2))-3.0f : 0.0f;
//     mp.skewness = (std>0) ? (m3/(std*std*std)) : 0.0f;
//     float thr = mean + 0.5f*std;
//     for (int i=1; i<n-1; i++)
//         if (mag[i]>mag[i-1] && mag[i]>mag[i+1] && mag[i]>thr) mp.peak_count++;
//     return mp;
// }

// ============================================================================
// CAPTURE: diagnostics + CIR_PWR register + CIR samples (in that order!)
// CIR_PWR register must be read before any RX reset.
// ============================================================================
static void capture_anchor_diagnostics(uint8_t anchor_idx)
{
    anchor_data_t *data = &anchor_data[anchor_idx];

    dwt_readdiagnostics(&data->diagnostics);

    // ★ Read CIR_PWR raw register IMMEDIATELY — cleared after dwt_rxreset()
    data->cir_pwr_raw = readCIR_PWR_raw();

    // FP_IDX: bits [15:6] of firstPath register give sample index
    data->fp_idx = (uint16_t)(data->diagnostics.firstPath >> 6);

    read_cir_samples(data->cir_samples, (int)data->fp_idx);

    // data->multipath = compute_multipath_features(data->cir_samples, CIR_SAMPLES_PER_ANCHOR);

    data->valid = 1;
}

// ============================================================================
// PRINT: one CSV row per anchor, eWINE-compatible columns 0–9 then CIR
//
// Python loading (config.py):
//   VNU_CIR_START   = 10     ← only difference from eWINE (15)
//   VNU_DIAG_COLS   = same as EWINE_DIAG_COLS (cols 1–9 identical)
// ============================================================================
static void print_all_distances(void)
{
    static char line_buffer[4096];

    complete_cycles++;

    // ── CSV header (printed once) ──────────────────────────────────────────
    if (!ml_header_printed)
    {
        printf("NLOS,RANGE,FP_IDX,FP_AMP1,FP_AMP2,FP_AMP3,"
               "STDEV_NOISE,CIR_PWR,MAX_NOISE,RXPACC");
        for (int s = 0; s < CIR_SAMPLES_PER_ANCHOR; s++)
            printf(",cir_%d", s);
        printf("\r\n");
        ml_header_printed = 1;
    }

    // ── One row per anchor ─────────────────────────────────────────────────
    for (int anchor_idx = 0; anchor_idx < NUM_ANCHORS; anchor_idx++)
    {
        anchor_data_t *data = &anchor_data[anchor_idx];
        if (!data->valid) continue;

        int pos = 0;
        dwt_rxdiag_t *d = &data->diagnostics;

        // Cols 0–9: label, diagnostics (match eWINE column positions exactly)
        // NOTE: distance stored as integer mm — %.2f dropped the decimal dot at
        // high baud (9126000), causing byte-drop corruption (787.98 → 78798).
        pos += sprintf(line_buffer + pos,
                       "$%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                       g_label,
                       (uint32_t)(data->distance + 0.5),  // col 1: RANGE (mm, int)
                       data->fp_idx,          // col 2: FP_IDX
                       d->firstPathAmp1,      // col 3: FP_AMP1
                       d->firstPathAmp2,      // col 4: FP_AMP2
                       d->firstPathAmp3,      // col 5: FP_AMP3
                       d->stdNoise,           // col 6: STDEV_NOISE
                       data->cir_pwr_raw,     // col 7: CIR_PWR (raw register)
                       d->maxNoise,           // col 8: MAX_NOISE
                       d->rxPreamCount);      // col 9: RXPACC

        // Cols 10–109: CIR magnitude
        for (int s = 0; s < CIR_SAMPLES_PER_ANCHOR; s++)
        {
            int32_t  r   = data->cir_samples[s].real;
            int32_t  im  = data->cir_samples[s].img;
            uint32_t mag = sqrt_uint32((uint32_t)(r * r + im * im));
            pos += sprintf(line_buffer + pos, ",%lu", mag);
        }

        // ── Unused columns from old format (kept as reference) ────────────
        // Old: anchor_id, maxGrowthCIR, fPathPWR_dBm, RX_PWR_dBm,
        //      kurtosis, skewness, peak_count
        // These are NOT part of eWINE cols 0–9 and NOT used by ECA-UWB.
        // ─────────────────────────────────────────────────────────────────

        line_buffer[pos++] = '\r';
        line_buffer[pos++] = '\n';
        line_buffer[pos]   = '\0';

        printf("%s", line_buffer);
    }
}

// ============================================================================
// MESSAGE UTILITIES
// ============================================================================
static void resp_msg_get_ts(uint8 *ts_field, uint32 *ts)
{
    *ts = 0;
    for (int i = 0; i < RESP_MSG_TS_LEN; i++)
        *ts += ts_field[i] << (i * 8);
}

static uint32 resp_msg_get_device_id(uint8 *field)
{
    uint32 id = 0;
    for (int i = 0; i < RESP_MSG_DEVICE_ID_LEN; i++)
        id += field[i] << (i * 8);
    return id;
}

static void poll_msg_set_device_id(uint8 *field, const uint32 device_id)
{
    for (int i = 0; i < POLL_MSG_DEVICE_ID_LEN; i++)
        field[i] = (device_id >> (i * 8)) & 0xFF;
}

// ============================================================================
// RANGING: SS-TWR to one anchor
// ============================================================================
static int ss_init_single_anchor(uint32 target_anchor_id, uint8_t anchor_idx)
{
    tx_poll_msg[ALL_MSG_SN_IDX] = frame_seq_nb;
    poll_msg_set_device_id(&tx_poll_msg[POLL_MSG_DEVICE_ID_IDX], target_anchor_id);

    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS);
    dwt_writetxdata(sizeof(tx_poll_msg), tx_poll_msg, 0);
    dwt_writetxfctrl(sizeof(tx_poll_msg), 0, 1);
    dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);
    tx_count++;

    TickType_t start_tick    = xTaskGetTickCount();
    TickType_t timeout_ticks = pdMS_TO_TICKS(RESPONSE_TIMEOUT_MS);

    while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) &
             (SYS_STATUS_RXFCG | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
    {
        if ((xTaskGetTickCount() - start_tick) > timeout_ticks)
        {
            dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
            dwt_rxreset();
            frame_seq_nb++;
            timeout_count++;
            return 0;
        }
        vTaskDelay(0);
    }

    frame_seq_nb++;

    if (status_reg & SYS_STATUS_RXFCG)
    {
        dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG);
        uint32 frame_len = dwt_read32bitreg(RX_FINFO_ID) & RX_FINFO_RXFLEN_MASK;

        if (frame_len <= RX_BUF_LEN)
            dwt_readrxdata(rx_buffer, frame_len, 0);

        rx_buffer[ALL_MSG_SN_IDX] = 0;

        if (memcmp(rx_buffer, rx_resp_msg, ALL_MSG_COMMON_LEN) == 0)
        {
            // ★ Capture diagnostics + CIR_PWR + CIR BEFORE any reset
            capture_anchor_diagnostics(anchor_idx);
            rx_count++;

            uint32 device_id = resp_msg_get_device_id(&rx_buffer[RESP_MSG_DEVICE_ID_IDX]);
            if (device_id != target_anchor_id) return 0;

            // TWR distance calculation
            uint32 poll_tx_ts, resp_rx_ts, poll_rx_ts, resp_tx_ts;
            poll_tx_ts = dwt_readtxtimestamplo32();
            resp_rx_ts = dwt_readrxtimestamplo32();

            float clockOffsetRatio =
                dwt_readcarrierintegrator() *
                (FREQ_OFFSET_MULTIPLIER * HERTZ_TO_PPM_MULTIPLIER_CHAN_5 / 1.0e6);

            resp_msg_get_ts(&rx_buffer[RESP_MSG_POLL_RX_TS_IDX], &poll_rx_ts);
            resp_msg_get_ts(&rx_buffer[RESP_MSG_RESP_TX_TS_IDX], &resp_tx_ts);

            int32 rtd_init = resp_rx_ts - poll_tx_ts;
            int32 rtd_resp = resp_tx_ts - poll_rx_ts;

            double tof = ((rtd_init - rtd_resp * (1.0f - clockOffsetRatio)) / 2.0f) * DWT_TIME_UNITS;
            anchor_data[anchor_idx].distance = tof * SPEED_OF_LIGHT * 1000.0;  // mm

            return 1;
        }
    }
    else
    {
        dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);
        dwt_rxreset();
    }

    return 0;
}

// ============================================================================
// MAIN RANGING + PRINT LOOP
// ============================================================================
int ss_init_run(void)
{
    for (int i = 0; i < NUM_ANCHORS; i++)
        anchor_data[i].valid = 0;

    for (int anchor_idx = 0; anchor_idx < NUM_ANCHORS; anchor_idx++)
    {
        TickType_t slot_start = xTaskGetTickCount();

        int success = ss_init_single_anchor(anchor_ids[anchor_idx], anchor_idx);
        if (!success) skip_count++;

        TickType_t elapsed    = xTaskGetTickCount() - slot_start;
        TickType_t slot_ticks = pdMS_TO_TICKS(TIME_SLOT_MS);
        if (elapsed < slot_ticks)
            vTaskDelay(slot_ticks - elapsed);
    }

    // Print 5 rows per cycle to increase collection speed
    for (int i = 0; i < 5; i++)
    {
        print_all_distances();
        vTaskDelay(pdMS_TO_TICKS(CYCLE_DELAY_MS));
    }
    vTaskDelay(pdMS_TO_TICKS(CYCLE_DELAY_MS));

    return 1;
}

// ============================================================================
// TASK ENTRY
// ============================================================================
void ss_initiator_task_function(void *pvParameter)
{
    UNUSED_PARAMETER(pvParameter);

    dwt_setleds(DWT_LEDS_ENABLE);

    printf("\r\n========================================\r\n");
    printf("ECA-UWB Data Collection v1.0\r\n");
    printf("Label: %s\r\n", g_label ? "NLOS (1)" : "LOS  (0)");
    printf("CIR window: FP_IDX-%d ... FP_IDX+%d (%d samples)\r\n",
           CIR_PRE_FP,
           CIR_SAMPLES_PER_ANCHOR - CIR_PRE_FP - 1,
           CIR_SAMPLES_PER_ANCHOR);
    printf("Anchor: 0x%04X\r\n", ANCHOR_1);
    printf("========================================\r\n\r\n");

    for (int i = 0; i < NUM_ANCHORS; i++)
        anchor_data[i].valid = 0;

    while (1)
        ss_init_run();
}
