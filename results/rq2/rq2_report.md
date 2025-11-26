# RQ2: Performance Archetypes
**Programs Analyzed**: sqlite, openssl, zstd, ffmpeg
**Total Paths**: 50047
**Discovered Archetypes**: 10

## Clustering Quality
- Silhouette Score: 0.247
- Calinski-Harabasz Index: 6835.0
- Davies-Bouldin Index: 1.405

## Discovered Archetypes

### Archetype 0
- **Size**: 9206 paths (18.4%)
- **Characteristics**: MEDIUM_DEPTH, CONCENTRATED
- **Resource Profile**: CPU_INTENSIVE
- **Dominant Type**: MIXED
- **Average Duration**: 40.49 ms
- **Average Path Length**: 13.7 functions
- **Application Distribution**:
  - zstd: 13 paths
  - ffmpeg: 9193 paths

### Archetype 1
- **Size**: 11602 paths (23.2%)
- **Characteristics**: MEDIUM_DEPTH
- **Resource Profile**: CPU_INTENSIVE
- **Dominant Type**: MIXED
- **Average Duration**: 1.57 ms
- **Average Path Length**: 13.0 functions
- **Application Distribution**:
  - sqlite: 7747 paths
  - openssl: 3646 paths
  - zstd: 63 paths
  - ffmpeg: 146 paths

### Archetype 2
- **Size**: 3164 paths (6.3%)
- **Characteristics**: SHALLOW
- **Resource Profile**: CPU_INTENSIVE
- **Dominant Type**: MIXED
- **Average Duration**: 0.33 ms
- **Average Path Length**: 7.2 functions
- **Application Distribution**:
  - sqlite: 391 paths
  - openssl: 1943 paths
  - zstd: 829 paths
  - ffmpeg: 1 paths

### Archetype 3
- **Size**: 2340 paths (4.7%)
- **Characteristics**: MEDIUM_DEPTH
- **Resource Profile**: CPU_INTENSIVE
- **Dominant Type**: MIXED
- **Average Duration**: 2.43 ms
- **Average Path Length**: 19.3 functions
- **Application Distribution**:
  - sqlite: 2340 paths

### Archetype 4
- **Size**: 5994 paths (12.0%)
- **Characteristics**: MEDIUM_DEPTH
- **Resource Profile**: CPU_INTENSIVE
- **Dominant Type**: MEMORY_BOUND
- **Average Duration**: 23.13 ms
- **Average Path Length**: 12.6 functions
- **Application Distribution**:
  - openssl: 65 paths
  - zstd: 4839 paths
  - ffmpeg: 1090 paths

### Archetype 5
- **Size**: 789 paths (1.6%)
- **Characteristics**: MEDIUM_DEPTH
- **Resource Profile**: CPU_INTENSIVE, IO_HEAVY
- **Dominant Type**: MIXED
- **Average Duration**: 15.95 ms
- **Average Path Length**: 14.5 functions
- **Application Distribution**:
  - sqlite: 32 paths
  - openssl: 671 paths
  - ffmpeg: 86 paths

### Archetype 6
- **Size**: 4403 paths (8.8%)
- **Characteristics**: MEDIUM_DEPTH
- **Resource Profile**: CPU_INTENSIVE
- **Dominant Type**: MIXED
- **Average Duration**: 0.56 ms
- **Average Path Length**: 19.5 functions
- **Application Distribution**:
  - sqlite: 1 paths
  - openssl: 4393 paths
  - ffmpeg: 9 paths

### Archetype 7
- **Size**: 3833 paths (7.7%)
- **Characteristics**: SHALLOW
- **Resource Profile**: CPU_INTENSIVE
- **Dominant Type**: MEMORY_BOUND
- **Average Duration**: 1.29 ms
- **Average Path Length**: 5.3 functions
- **Application Distribution**:
  - sqlite: 3446 paths
  - openssl: 330 paths
  - zstd: 32 paths
  - ffmpeg: 25 paths

### Archetype 8
- **Size**: 1086 paths (2.2%)
- **Characteristics**: MEDIUM_DEPTH
- **Resource Profile**: CPU_INTENSIVE, MEMORY_GROWING
- **Dominant Type**: MIXED
- **Average Duration**: 6.60 ms
- **Average Path Length**: 11.4 functions
- **Application Distribution**:
  - sqlite: 1043 paths
  - openssl: 40 paths
  - zstd: 3 paths

### Archetype 9
- **Size**: 7630 paths (15.2%)
- **Characteristics**: SHALLOW
- **Resource Profile**: CPU_INTENSIVE
- **Dominant Type**: MIXED
- **Average Duration**: 37.12 ms
- **Average Path Length**: 5.6 functions
- **Application Distribution**:
  - zstd: 3190 paths
  - ffmpeg: 4440 paths

## Stability Analysis
- **Iteration Stability**: 60.7%

### Program Diversity
- **sqlite**: 5 unique archetypes (diversity: 0.50)
- **openssl**: 5 unique archetypes (diversity: 0.50)
- **zstd**: 4 unique archetypes (diversity: 0.40)
- **ffmpeg**: 4 unique archetypes (diversity: 0.40)

## Universal Patterns

### Archetype_0
- **Universality**: 50.0% of applications
- **Instances**: 9206
- **Characteristics**: MEDIUM_DEPTH, CONCENTRATED

### Archetype_1
- **Universality**: 100.0% of applications
- **Instances**: 11602
- **Characteristics**: MEDIUM_DEPTH

### Archetype_2
- **Universality**: 100.0% of applications
- **Instances**: 3164
- **Characteristics**: SHALLOW

### Archetype_4
- **Universality**: 75.0% of applications
- **Instances**: 5994
- **Characteristics**: MEDIUM_DEPTH

### Archetype_5
- **Universality**: 75.0% of applications
- **Instances**: 789
- **Characteristics**: MEDIUM_DEPTH

### Archetype_6
- **Universality**: 75.0% of applications
- **Instances**: 4403
- **Characteristics**: MEDIUM_DEPTH

### Archetype_7
- **Universality**: 100.0% of applications
- **Instances**: 3833
- **Characteristics**: SHALLOW

### Archetype_8
- **Universality**: 75.0% of applications
- **Instances**: 1086
- **Characteristics**: MEDIUM_DEPTH

### Archetype_9
- **Universality**: 50.0% of applications
- **Instances**: 7630
- **Characteristics**: SHALLOW
