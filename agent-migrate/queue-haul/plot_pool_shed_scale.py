"""Plot the retained supplemental fleet/WAN comparison."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import plot_style as style

out = Path(__file__).resolve().parent / 'outputs/a100-pooled-service'
source = out / 'scale-diagnostic.json'
data = json.loads(source.read_text())
assert data['complete'] and data['pass'] and len(data['rows']) == 24
assert hashlib.sha256((out / data['producer']).read_bytes()).hexdigest() == data['producer_sha256']
style.apply()
modes = ('shared_baseline', 'fixed_wan', 'proportional_wan')
labels = ('20 MW\n1000 Gbit/s', '2 MW\n1000 Gbit/s', '2 MW\n100 Gbit/s')
figures = []
for deadline in (30, 300):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout='constrained')
    for column, (workload, load) in enumerate((('coding', .5), ('coding', .95), ('coding_long', .5), ('coding_long', .95))):
        rows = {r['wan_mode']: r for r in data['rows'] if (r['workload'], r['resident_load'], r['deadline_s']) == (workload, load, deadline)}
        assert set(rows) == set(modes)
        for policy in ('queue_haul', 'replay_only'):
            values = [100 * rows[m]['policies'][policy]['shed_fraction'] for m in modes]
            axes[0, column].plot(range(3), values, label=style.POLICY_NAMES[policy], color=style.POLICY_COLORS[policy], linestyle=style.POLICY_LINESTYLES[policy], marker=style.POLICY_MARKERS[policy], linewidth=2)
        kv = [100 * rows[m]['policies']['queue_haul']['kv_shed_fraction'] for m in modes]
        axes[1, column].bar(range(3), kv, color=style.ACTION_COLORS['kv_transfer'], width=.55)
        for x, value in enumerate(kv):
            axes[1, column].annotate(f'{value:.2f}', (x, value), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
        axes[0, column].set_title(f'{"Coding" if workload == "coding" else "Long coding"} · {load:.0%} resident load', fontsize=11)
        for ax in axes[:, column]:
            ax.set_xticks(range(3), labels, fontsize=9)
            ax.tick_params(axis='y', labelsize=10)
            ax.set_xlim(-.5, 2.5)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        axes[0, column].set_ylim(0, 105 if load == .5 else 13.2)
        axes[1, column].set_ylim(0, max(1., max(kv) * 1.22))
    axes[0, 0].set_ylabel('Source handoff (%)', fontsize=11)
    axes[1, 0].set_ylabel('QH completed KV\n(% of original source)', fontsize=11)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncols=2, fontsize=10, title='Fleet labels are GPU nameplate; WAN is a scenario allocation, not measured region-pair capacity.', title_fontsize=8)
    fig.suptitle(f"{deadline} s deadline · {style.MODEL_NAMES['openai/gpt-oss-20b']} on {style.AGENTIC_HARDWARE_NAMES['a100']} · three equal-size sites", fontsize=14)
    for suffix in ('png', 'pdf'):
        path = out / f'scale-comparison-{deadline}s.{suffix}'
        fig.savefig(path, dpi=style.SAVE_DPI)
        figures.append(path.name)
    plt.close(fig)
(out / 'scale-figure-provenance.json').write_text(json.dumps({'identity': data['identity'], 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'figures': figures, 'scope': 'Central calibration; all 24 existing cells retained. KV percentages use the entire original source, not only completed handoffs.'}, sort_keys=True) + '\n')
