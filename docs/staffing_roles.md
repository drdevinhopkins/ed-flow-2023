# L1/L2 scheduled deployment

Devin Hopkins confirmed the new L2 arrangement began **October 1, 2026**.

| Schedule situation | L1 | L2 |
| --- | --- | --- |
| Before October 1, 2026 | Historical overlap classification | Historical overlap classification; four-hour training or return-to-practice shifts |
| October 1 onward, Mon–Thu, simultaneous L2 | Vertical / ambulatory | Stretcher / POD |
| October 1 onward, no simultaneous L2 | Flexible, where needed | Absent |
| Friday / weekend with an unexpected L2 | Flexible | Preserve overlap; no zone rule was confirmed |

`scripts/staffing_roles.py` resolves each active hour using the effective date and actual simultaneous schedule entries. L2 is not imputed from the weekday. Raw shift codes and physician identities remain intact. For example, today's export lists L1 12:00–21:00 and L2 13:00–21:00: L1 is flexible at noon and Vertical from 13:00 through 20:00. Shift intervals exclude the end hour.

The rule feeds shared staffing counts, physician-role features, structural coverage changes and physician-effect fitting/scoring, as well as on-call probability and impact features. The `flexible` role is separate from `overlap`; `n_l1_vertical`, `n_l1_flexible` and `n_l2_pod` provide explicit schedule-derived counts. Actual shift starts/ends remain personnel events even if the deployment changes mid-shift. Historical roles before the effective date remain unchanged; new numeric columns are zero there.

The on-call model training version includes the staffing rule version, forcing retraining of incompatible cached models. These changes describe scheduled capacity, not proof of attendance or on-call availability, and do not establish a causal effect of L2 on flow. A new deployment needs prospective validation; no forecast accuracy improvement is claimed.

Blurb facts expose the L1/L2 assignment from the optional schedule export. With the split active, deterministic and LLM blurbs preserve L1 on Vertical and L2 on POD. If the schedule is unavailable, the blurb asks to assess available coverage without promising a flexible L1.
