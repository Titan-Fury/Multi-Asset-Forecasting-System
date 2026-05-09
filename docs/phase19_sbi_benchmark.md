# Phase-19 SBI benchmark

## Pipeline parity

```
                    variant  proba_up  blended_p  direction  conviction  elapsed_s
baseline (emb+microstr OFF)    0.5284      0.519          0         3.8      20.15
       with new features ON    0.5284      0.519          0         3.8      21.13
```

proba_up delta = `0.0000`

## Feature information value (vs fwd_ret_5d)

```
           feature    n  corr_fwd_ret_5d  univariate_auc  auc_minus_05
delivery_pct_z_60d    0              NaN             NaN           NaN
  oi_change_pct_5d    0              NaN             NaN           NaN
 fii_dii_net_z_20d    0              NaN             NaN           NaN
  amihud_illiq_20d 1228           0.0779          0.5079        0.0079
  signed_vol_proxy    0              NaN             NaN           NaN
    microstr_score    0              NaN             NaN           NaN
            emb_00 1163              NaN             NaN           NaN
            emb_01 1163              NaN             NaN           NaN
            emb_02 1163              NaN             NaN           NaN
            emb_03 1163              NaN             NaN           NaN
```

## Speed

- baseline: 20.15s
- with new: 21.13s (overhead +0.98s)
