#!/usr/bin/env python3
"""
MCvD-Transformer Inference Script
--------------------------------
This script performs inference using a trained MCvD-Transformer model.
It can visualize the Channel Impulse Response (CIR) with or without ground truth data.

Usage:
    python inference.py --model path/to/model.keras --topology path/to/topology.json [--ground-truth path/to/time_output.npy]

Authors: [Your Name]
Date: December 2024
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

from mcvd_transformer.utils.objects import Topology, CoordinateSystem
from mcvd_transformer.utils.postprocessing import PostProcessing
from mcvd_transformer.model.layers import TransformerBlock
from mcvd_transformer.model.metrics import (
    r2_score,
    max_loss,
    custom_mse,
    custom_mape,
    max_squared_loss
)
from mcvd_transformer.utils.evaluator import PerformanceEvaluator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform inference using a trained MCvD-Transformer model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model file (.keras)'
    )
    
    parser.add_argument(
        '--topology',
        type=str,
        required=True,
        help='Path to the topology configuration file (topology.json)'
    )
    
    parser.add_argument(
        '--ground-truth',
        type=str,
        help='Path to the ground truth time output file (time_output.npy)'
    )
    
    parser.add_argument(
        '--intended-index',
        type=int,
        default=0,
        help='Index of the intended absorber in the topology'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save visualization results'
    )
    
    parser.add_argument(
        '--save-plot',
        action='store_true',
        help='Save the visualization plot instead of displaying it'
    )
    
    return parser.parse_args()

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load the saved model with custom objects properly registered.
    
    Args:
        model_path: Path to the saved Keras model
        
    Returns:
        Loaded model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is invalid
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    custom_objects = {
        'TransformerBlock': TransformerBlock,
        'r2_score': r2_score,
        'max_loss': max_loss,
        'custom_mse': custom_mse,
        'custom_mape': custom_mape,
        'max_squared_loss': max_squared_loss
    }
    
    try:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def predict_cir(
    model: tf.keras.Model,
    topology: Topology,
    rotation_theta: float = 0,
    rotation_phi: float = 0,
    coordinate_system: CoordinateSystem = CoordinateSystem.BOTH,
    max_spherical_entity: int = 15,
    entity_order: str = "shuffle"
) -> tuple:
    """
    Makes inference using the trained MCvD Transformer model.
    
    Args:
        model: Trained MCvD Transformer model
        topology: MCvD channel topology instance
        rotation_theta: Rotation angle theta in degrees [0, 360)
        rotation_phi: Rotation angle phi in degrees [-90, 90]
        coordinate_system: Type of coordinate system to use
        max_spherical_entity: Maximum number of spherical entities
        entity_order: Order of entities ("shuffle" or "normal")
        
    Returns:
        Tuple containing:
        - predicted_cir: Channel impulse response signal
        - shape_error: Mean absolute error of the predicted shape (if ground truth available)
        - max_error: Relative error of the predicted maximum value (if ground truth available)
    """
    if rotation_theta != 0 or rotation_phi != 0:
        topology = topology.rotate(rotation_theta, rotation_phi)
    
    input_top, input_num = topology.convert_numpy(
        coordinate_system=coordinate_system,
        max_spherical_entity=max_spherical_entity,
        order=entity_order,
        flatten=False,
        one_absorber_points=4
    )
    
    shape, max_value, _ = model.predict([
        np.expand_dims(input_top, axis=0),
        np.expand_dims(input_num, axis=0)
    ])
    
    predicted_cir = PostProcessing.postprocessing_separate(shape[0], max_value[0])
    
    shape_error = None
    max_error = None
    if hasattr(topology, 'time_output'):
        predicted_shape = predicted_cir / predicted_cir[-1]
        actual_shape = topology.time_output / topology.time_output[-1]
        shape_error = float(np.mean(np.abs(predicted_shape - actual_shape)))
        max_error = float(np.abs(max_value - topology.time_output[-1]) / topology.time_output[-1])
    
    return predicted_cir, shape_error, max_error

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create output directory if saving results
    if args.save_plot:
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load model
        print("Loading model...")
        model = load_model(args.model)
        print("Model loaded successfully")
        
        # Load topology
        print("Loading topology...")
        topology = Topology.read_config(args.topology, args.intended_index)
        
        # Load ground truth if provided
        if args.ground_truth and os.path.exists(args.ground_truth):
            print("Loading ground truth data...")
            ground_truth = np.load(args.ground_truth)
            topology.set_time_output(ground_truth[args.intended_index])
            print("Ground truth data loaded successfully")
        
        # Make prediction
        print("Making prediction...")
        cir, shape_err, max_err = predict_cir(model, topology)
        print("Prediction successful")
        
        # Print error metrics if ground truth was provided
        if shape_err is not None:
            print(f"Shape error: {shape_err:.4f}")
            print(f"Max error: {max_err:.4f}")
        
        # Visualize results
        print("Generating visualization...")
        plot_path = os.path.join(args.output_dir, 'cir_plot.png') if args.save_plot else None
        
        PerformanceEvaluator.time_graph(
            time_output_actual=topology.time_output if hasattr(topology, 'time_output') else np.zeros_like(cir),
            time_output_predicted_list=[cir],
            legend=["Ground Truth", "Prediction"] if hasattr(topology, 'time_output') else ["Prediction"],
            time_res=1,
            expension_ratio=1,
            path=plot_path
        )
        
        if args.save_plot:
            print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()