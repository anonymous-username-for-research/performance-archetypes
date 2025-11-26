# RQ1: The Static-Dynamic Paradox
**Programs Analyzed**: sqlite, openssl, zstd, ffmpeg

## Detailed Results

### OPENSSL

#### Paradox Analysis
- **Simple but Critical**: 4 functions
- **Complex but Irrelevant**: 1 functions
- **Matched Expectations**: 103 functions

**Top Simple but Critical Functions:**
- `BIO_new`: static rank #15305 → dynamic rank #4
- `CRYPTO_new_ex_data`: static rank #15228 → dynamic rank #13
- `OBJ_NAME_init`: static rank #14917 → dynamic rank #35
- `ossl_property_name`: static rank #13893 → dynamic rank #36

#### Hidden Bottlenecks
Found 8 hidden bottlenecks:
- `BIO_new`: 1 LOC, impact score: 4.49
- `CRYPTO_new_ex_data`: 1 LOC, impact score: 4.15
- `dup_bio_in`: 1 LOC, impact score: 3.48
- `ossl_lib_ctx_get_concrete`: 3 LOC, impact score: 3.00
- `ossl_lib_ctx_get_ex_data_global`: 4 LOC, impact score: 2.87

#### Static-Dynamic Correlation
- Pearson r = 0.511 (p = 0.0000)
- Spearman ρ = 0.543 (p = 0.0000)

### FFMPEG

#### Paradox Analysis
- **Simple but Critical**: 189 functions
- **Complex but Irrelevant**: 188 functions
- **Matched Expectations**: 2238 functions

**Top Simple but Critical Functions:**
- `ff_thread_await_progress`: static rank #25014 → dynamic rank #93
- `avpriv_slicethread_free`: static rank #24162 → dynamic rank #62
- `ffurl_close`: static rank #23440 → dynamic rank #12
- `just_return`: static rank #25006 → dynamic rank #270
- `avcodec_find_encoder`: static rank #24191 → dynamic rank #127

#### Hidden Bottlenecks
Found 5 hidden bottlenecks:
- `ffurl_close`: 1 LOC, impact score: 5.10
- `avio_closep`: 3 LOC, impact score: 4.29
- `file_close`: 3 LOC, impact score: 3.12
- `ff_thread_free`: 4 LOC, impact score: 2.95
- `sch_wait`: 12 LOC, impact score: 2.87

#### Static-Dynamic Correlation
- Pearson r = 0.192 (p = 0.0000)
- Spearman ρ = 0.175 (p = 0.0000)

### SQLITE

#### Paradox Analysis
- **Simple but Critical**: 130 functions
- **Complex but Irrelevant**: 178 functions
- **Matched Expectations**: 1239 functions

**Top Simple but Critical Functions:**
- `sqlite3_open_v2`: static rank #5464 → dynamic rank #5
- `sqlite3PagerIsreadonly`: static rank #5862 → dynamic rank #81
- `sqlite3PcacheRefCount`: static rank #5854 → dynamic rank #80
- `sqlite3_create_function`: static rank #5472 → dynamic rank #33
- `sqlite3_create_collation`: static rank #5473 → dynamic rank #34

#### Hidden Bottlenecks
Found 3 hidden bottlenecks:
- `sqlite3_open_v2`: 1 LOC, impact score: 4.23
- `sqlite3_prepare_v2`: 4 LOC, impact score: 2.86
- `sqlite3_stmtrand_init`: 7 LOC, impact score: 2.12

#### Static-Dynamic Correlation
- Pearson r = 0.062 (p = 0.0151)
- Spearman ρ = -0.041 (p = 0.1035)

### ZSTD

#### Paradox Analysis
- **Simple but Critical**: 31 functions
- **Complex but Irrelevant**: 33 functions
- **Matched Expectations**: 501 functions

**Top Simple but Critical Functions:**
- `UTIL_countCores`: static rank #2563 → dynamic rank #55
- `HUF_compress1X_usingCTable_internal_bmi2`: static rank #2506 → dynamic rank #51
- `AIO_IOPool_threadPoolActive`: static rank #2737 → dynamic rank #133
- `UTIL_countLogicalCores`: static rank #2473 → dynamic rank #56
- `ZSTD_entropyCompressSeqStore`: static rank #2316 → dynamic rank #36

#### Hidden Bottlenecks
Found 3 hidden bottlenecks:
- `ZSTD_compressContinue_public`: 2 LOC, impact score: 4.60
- `FIO_compressFilename`: 4 LOC, impact score: 3.75
- `FIO_decompressFilename`: 4 LOC, impact score: 3.66

#### Static-Dynamic Correlation
- Pearson r = 0.382 (p = 0.0000)
- Spearman ρ = 0.336 (p = 0.0000)