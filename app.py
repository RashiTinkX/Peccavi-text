# app.py
import gradio as gr

from peccavi.eval import run_evaluation, refresh_baselines


def build_ui():
    with gr.Blocks(title="PECCAVI-TEXT") as demo:

        gr.Markdown("""
# PECCAVI-TEXT — Evaluation Dashboard
**Watermarking & Content Authenticity**
*Tournament Sampling · Policy Learning over Simulated Generations*

---
Enter a prompt. PECCAVI watermarks it, Scriba attacks it with 5 adversarial paraphrases,
Custos detects the watermark, then results are compared against four baseline model families.
""")

        #  Input row
        with gr.Row():
            with gr.Column(scale=4):
                prompt_box = gr.Textbox(
                    label="Input Prompt",
                    placeholder="e.g. Explain the importance of AI safety in modern systems.",
                    lines=3,
                )
            with gr.Column(scale=1):
                theta_slider = gr.Slider(
                    minimum=0.1, maximum=10.0, value=2.0, step=0.1,
                    label="Watermark Strength (θ)",
                    info="Higher θ = stronger watermark signal",
                )

        with gr.Row():
            run_btn = gr.Button("▶  Run PECCAVI Evaluation", variant="primary", size="lg")
            refresh_btn = gr.Button("⟳ Refresh Baseline Results", variant="secondary")

        #  Side-by-side outputs
        gr.Markdown("### Generated Text")
        with gr.Row():
            plain_out = gr.Textbox(
                label="Plain LLM Output  (backbone only, no watermark)",
                lines=6, max_lines=8,
            )
            wm_out = gr.Textbox(
                label="PECCAVI Watermarked Output  (Auctor — tournament sampling)",
                lines=6, max_lines=8,
            )

        #  Detection
        gr.Markdown("### Watermark Detection  (Custos)")
        with gr.Row():
            det_out = gr.Textbox(label="Detection Result & Score", scale=3)
            seff_out = gr.Textbox(label="Effective Score  S_eff  (worst-case across paraphrases)", scale=2)

        # Paraphrase robustness
        gr.Markdown("### Adversarial Paraphrase Test  (Scriba)")
        paraphrase_out = gr.Textbox(
            label="5 adversarial variants — per-variant watermark score",
            lines=12, max_lines=14,
        )

        #  Comparison table
        gr.Markdown("""
Baseline Comparison

PECCAVI- metrics are computed **live** from the prompt above.
Baseline values are pre-computed reference benchmarks representing standard (non-PECCAVI)
watermarking applied to each model family (GPT-4, Claude-3, DeepSeek, LLaMA-2).
""")
        table_out = gr.Textbox(
            label="Metrics Table",
            lines=10, max_lines=10,
        )
        state_peccavi = gr.State()

        # Plots
        gr.Markdown("Evaluation Plots")
        with gr.Row():
            bar_plot = gr.Plot(label="Grouped Bar Chart — Metrics vs All Baselines")
            radar_plot = gr.Plot(label="Radar Chart — Normalised Overall Performance")

        run_btn.click(
            fn=run_evaluation,
            inputs=[prompt_box, theta_slider],
            outputs=[
                plain_out, wm_out,
                det_out, seff_out, paraphrase_out,
                table_out,
                bar_plot, radar_plot,
                state_peccavi,
            ],
        )

        refresh_btn.click(
            fn=refresh_baselines,
            inputs=[state_peccavi],
            outputs=[table_out, bar_plot, radar_plot],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=True, theme=gr.themes.Soft())

