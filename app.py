import gradio as gr

from data.loaders import generate_synthetic_data, load_swiss_radar_data
from evaluation.metrics import compute_metrics, print_comparison
from evaluation.visualization import create_loss_plot, create_prediction_visualization
from models.linda_pinn import LINDAPINNModel, device
from training.trainers import (
    LINDAPINNTrainer,
    run_comparison,
    train_custom_pinn,
    train_custom_pinn_with_params,
    train_traditional_linda,
    train_traditional_linda_with_params,
)


def create_gradio_app():
    with gr.Blocks(title="LINDA vs LINDA-PINN Comparison") as app:
        gr.Markdown("""
        # LINDA vs LINDA-PINN Weather Nowcasting Comparison
        
        Compare traditional LINDA with Physics-Informed Neural Network (PINN) approach for precipitation nowcasting.
        Adjust hyperparameters for both models and see how they perform!
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### LINDA Parameters")
                gr.Markdown("#### Data Configuration")
                linda_n_input = gr.Slider(3, 10, value=3, step=1, label="Input Frames (n_input)")
                linda_n_forecast = gr.Slider(1, 256, value=6, step=1, label="Forecast Frames (n_forecast)")

                gr.Markdown("#### Model Parameters")
                linda_n_ens = gr.Slider(1, 50, value=10, step=1, label="Ensemble Members")
                linda_vel_p1 = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Velocity Perturbation P1")
                linda_vel_p2 = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Velocity Perturbation P2")
                linda_vel_p3 = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Velocity Perturbation P3")
                linda_vel_p4 = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Velocity Perturbation P4")
                linda_vel_p5 = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Velocity Perturbation P5")
                linda_km = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="KM per Pixel")
                linda_timestep = gr.Slider(1, 15, value=5, step=1, label="Timestep (minutes)")

            with gr.Column():
                gr.Markdown("### PINN Parameters")
                gr.Markdown("#### Data Configuration")
                pinn_n_input = gr.Slider(3, 10, value=3, step=1, label="Input Frames (n_input)")
                pinn_n_forecast = gr.Slider(1, 256, value=3, step=1, label="Forecast Frames (n_forecast)")

                gr.Markdown("#### Model Parameters")
                pinn_epochs = gr.Slider(5, 100, value=10, step=5, label="Training Epochs")
                pinn_lr = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning Rate")
                pinn_weight_decay = gr.Slider(1e-6, 1e-3, value=1e-5, step=1e-6, label="Weight Decay")
                pinn_hidden = gr.Slider(64, 512, value=256, step=64, label="Hidden Layer Size")
                pinn_layers = gr.Slider(2, 8, value=5, step=1, label="Number of Layers")
                pinn_sigma = gr.Slider(-2.0, 2.0, value=0.0, step=0.1, label="Initial Log Sigma")
                pinn_survival = gr.Slider(0.1, 1.0, value=0.8, step=0.1, label="Initial Survival Probability")
                pinn_growth = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Initial Growth Rate")

        with gr.Row():
            use_synthetic = gr.Checkbox(value=True, label="Use Synthetic Data (faster)")
            run_btn = gr.Button("Run Comparison", variant="primary")

        with gr.Row():
            results_output = gr.Markdown()

        with gr.Row():
            predictions_plot = gr.Plot(label="Predictions Comparison")
            loss_plot = gr.Plot(label="PINN Training Loss")

        run_btn.click(
            fn=run_comparison,
            inputs=[
                linda_n_input,
                linda_n_forecast,
                linda_n_ens,
                linda_vel_p1,
                linda_vel_p2,
                linda_vel_p3,
                linda_vel_p4,
                linda_vel_p5,
                linda_km,
                linda_timestep,
                pinn_n_input,
                pinn_n_forecast,
                pinn_epochs,
                pinn_lr,
                pinn_weight_decay,
                pinn_hidden,
                pinn_layers,
                pinn_sigma,
                pinn_survival,
                pinn_growth,
                use_synthetic,
            ],
            outputs=[results_output, predictions_plot, loss_plot],
        )

        gr.Markdown("""
        ### About
        - **LINDA**: Lagrangian Integro-Difference equation with Nowcasting and Data Assimilation
        - **PINN/LINDA-PINN**: LINDA-inspired integro-difference PINN model
        - **n_input**: Number of past frames used to make predictions
        - **n_forecast**: Number of future frames to predict
        - Metrics shown are computed on test data
        """)

    return app


# Launch the app
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)
