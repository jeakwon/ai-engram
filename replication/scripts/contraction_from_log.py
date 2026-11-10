"""Rebuild the result JSON of a contraction_response.py run from its log.

Runs 25246271 and 25246272 printed every measurement, then stopped at a control that compared the first-order term
of the Hessian-vector pass (batch 4) with the stored one (batch 8) too tightly, before writing their JSON. This
script reads the printed lines back: the update norm and g^T Delta, every grid row (displacement, losses, their
change and first-order prediction, KL to Init, the six scores), the finite differences, the Hessian-vector
curvatures and the controls, and adds the second-order prediction and the remainders as contraction_response.py
would. Printed precision applies (losses 4 decimals, first-order terms 5, scores 4).

Positive control: the grid has every t of ``--t_grid`` and t = 0 has zero change, or the script stops.

Usage:
    python replication/scripts/contraction_from_log.py --log <slurm log> --out_json <file>
"""

import argparse
import json
import re

NUM = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--t_grid", type=float, nargs="+", required=True, help="the run's --t_grid (the log does not print it)")
    args = p.parse_args()
    text = open(args.log).read()
    out = {"source": args.log, "controls": [], "grid": [], "finite_differences": [], "hvp": {}}
    d = re.search(rf"\[direction\] (\w+): \|\|Delta W\|\| ({NUM}) \((\{{.*?\}})\); g_f\^T Delta ({NUM}), g_r\^T Delta ({NUM})", text)
    if not d:
        raise SystemExit("no [direction] line")
    out.update({"direction": d.group(1), "update_norm": float(d.group(2)), "update_norm_by_type": json.loads(d.group(3)),
                "g_dot_delta": {"forget": float(d.group(4)), "retain": float(d.group(5))}})
    curve = re.compile(rf"\[curve\] t ({NUM}): displacement ({NUM}); loss forget ({NUM}) \(({NUM}), first order ({NUM})\), "
                       rf"retain ({NUM}) \(({NUM}), ({NUM})\); KL forget ({NUM}) retain ({NUM})(?:; (.*))?")
    for line in text.splitlines():
        m = curve.match(line)
        if m:
            v = [float(m.group(i)) for i in range(1, 11)]
            row = {"t": v[0], "displacement": v[1], "loss": {"forget": v[2], "retain": v[5]},
                   "change": {"forget": v[3], "retain": v[6]}, "first_order": {"forget": v[4], "retain": v[7]},
                   "kl_to_init": {"forget": v[8], "retain_perturbed": v[9]}}
            if m.group(11):
                row["scores"] = {k: float(x) for k, x in re.findall(rf"(\w+)=({NUM})", m.group(11))}
            out["grid"].append(row)
        m = re.match(rf"\[fd\] h ({NUM}): forget phi' ({NUM}) phi'' ({NUM}); retain phi' ({NUM}) phi'' ({NUM})", line)
        if m:
            h, f1, f2, r1, r2 = (float(m.group(i)) for i in range(1, 6))
            out["finite_differences"].append({"h": h, "forget": {"first": f1, "second": f2},
                                              "retain": {"first": r1, "second": r2}})
        m = re.match(rf"\[hvp\] (\w+): Delta\^T H Delta ({NUM}), g\^T Delta in the same pass ({NUM})", line)
        if m:
            out["hvp"][m.group(1)] = {"curvature": float(m.group(2)), "first": float(m.group(3))}
        if line.startswith("[control]"):
            out["controls"].append(line)
    out["grid"].sort(key=lambda r: r["t"])
    expected = sorted(args.t_grid)
    got = [r["t"] for r in out["grid"]]
    if len(got) != len(expected) or any(abs(a - b) > 1e-9 for a, b in zip(got, expected)):
        raise SystemExit(f"grid {got} does not match the run's t grid {expected}")
    if got[0] == 0.0 and any(abs(out["grid"][0]["change"][s]) > 1e-9 for s in ("forget", "retain")):
        raise SystemExit("t = 0 shows a loss change")
    out["loss_init"] = out["grid"][0]["loss"] if got[0] == 0.0 else None
    for r in out["grid"]:
        t = r["t"]
        if out["hvp"]:
            r["second_order"] = {s: t * out["g_dot_delta"][s] + 0.5 * t * t * out["hvp"][s]["curvature"]
                                 for s in out["hvp"]}
            r["remainder_second"] = {s: r["change"][s] - r["second_order"][s] for s in out["hvp"]}
        r["remainder_first"] = {s: r["change"][s] - r["first_order"][s] for s in ("forget", "retain")}
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {args.out_json}: {len(out['grid'])} grid rows, {len(out['finite_differences'])} finite-difference "
          f"steps, hvp {sorted(out['hvp'])}, {len(out['controls'])} control lines")


if __name__ == "__main__":
    main()
