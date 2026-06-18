## 1000 per split, MU = 20, NUM_GHOSTS = 50

AUC for TM vs NT: 0.9650
AUC for T vs NT: 0.8668
T mean loss:  2.3413
TM mean loss: 1.9909
NT mean loss: 3.0440 

## 10K per split, 500 * 20

AUC for TM vs NT: 0.9463                                                                                               │
AUC for T vs NT: 0.8451                                                                                                │
T mean loss:  2.4272                                                                                                   │
TM mean loss: 2.1390                                                                                                   │
NT mean loss: 2.9826                                                                                                   │

## First result

Unique ghosts for every document (10k)
    - Full document MIA remains high, not caused mainly by ghosts
    - Ghost-only MIA is weak but above random
    - Appending gives a very slight stronger ghost-only signal than prepending

### Appended 1x10000
    MT ghosts: 10000
    MNT ghosts: 10000
    AUC for TM vs NT: 0.8025
    AUC for T vs NT: 0.8449
    AUC for TM vs NTM: 0.8314
    T mean loss: 2.4267
    TM mean loss: 2.5208
    NT mean loss: 2.9831
    NTM mean loss: 3.0130
    Ghost-only AUC for TM vs NTM: 0.5880

### Prepended 1x10000
    TM ghosts: 10000 
    NTM ghosts: 10000 
    AUC for TM vs NT: 0.8098
    AUC for T vs NT: 0.8457 
    AUC for TM vs NTM: 0.8353 
    T mean loss: 2.4279 
    TM mean loss: 2.5192 
    NT mean loss: 2.9881
    NTM mean loss: 3.0007 
    Ghost-only AUC for TM vs NTM: 0.5627

### Appended 5x2000
    MT ghosts: 2000
    MNT ghosts: 2000
    AUC for TM vs NT: 0.9496
    AUC for T vs NT: 0.8422
    AUC for TM vs NTM: 0.9778
    T mean loss: 2.4326
    TM mean loss: 2.1371
    NT mean loss: 2.9832
    NTM mean loss: 3.1930
    Ghost-only AUC for TM vs NTM: 1.0000

    # Conclusion -> Single-occurrence unique ghosts produce only weak span-specific membership evidence, while repeated ghost identifiers produce a strong loss-based membership signal.

