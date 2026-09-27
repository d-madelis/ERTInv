# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 01:49:59 2025

@author: madel
"""


"""
ERT Inversion and Residual Analysis
===================================
This script performs synthetic ERT modeling with complex geological structures,
runs inversion with different algorithms and noise levels, and conducts 
comprehensive residual analysis.

Author: Your Name
Date: 2025
License: MIT
"""

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm


def create_geometry():
    """
    Create complex geological geometry with three horizontal layers 
    and irregular shaped targets.
    
    Returns:
        geom: PyGIMLI geometry object
    """
    # 1. Create world with three horizontal layers
    world = mt.createWorld(start=[-40, 0], end=[40, -50], 
                          layers=[-2, -8], worldMarker=True)
    
    # 2. Irregular targets
    # Target 1: Irregular polygon (high resistivity)
    target1 = mt.createPolygon([
        [-17, -4], [-12, -3], [-8, -6], [-13, -8], [-16, -7]
    ], isClosed=True, marker=4, area=0.3)
    
    # Target 2: Irregular polygon (low resistivity)  
    target2 = mt.createPolygon([
        [3, -5], [7, -4], [9, -7], [6, -9], [4, -8]
    ], isClosed=True, marker=5, area=0.3)
    
    # 3. Large irregular shallow target (very high resistivity)
    large_target = mt.createPolygon([
        [-20, -8], [-5, -10], [10, -8], [20, -14], [15, -20], [-10, -18], [-20, -14]
    ], isClosed=True, marker=6, area=0.3)
    
    # 4. Merge all geometries
    geom = world + target1 + target2 + large_target
    
    return geom


def setup_measurement_scheme(geom):
    """
    Set up electrode configuration and measurement scheme.
    
    Args:
        geom: PyGIMLI geometry object
        
    Returns:
        scheme: ERT measurement scheme
    """
    # Create 31 electrodes linearly spaced from -20m to 20m
    elecs = np.linspace(start=-20, stop=20, num=31)
    scheme = ert.createData(elecs=elecs, schemeName='dd')
    
    # Add electrode positions to geometry
    for p in scheme.sensors():
        geom.createNode(p)
    
    return scheme


def create_resistivity_map():
    """
    Define resistivity values for different geological units.
    
    Returns:
        rhomap: List of [marker, resistivity] pairs
    """
    rhomap = [
        [0, 100.],   # Surface layer
        [1, 80.],    # Shallow layer
        [2, 30.],    # Middle layer
        [3, 200.],   # Deep layer
        [4, 400.],   # Target 1 - High resistivity
        [5, 10.],    # Target 2 - Low resistivity
        [6, 500.]    # Large target - Very high resistivity
    ]
    return rhomap


def generate_synthetic_datasets(mesh, scheme, rhomap, noise_levels):
    """
    Generate synthetic ERT datasets with different noise levels.
    
    Args:
        mesh: Computational mesh
        scheme: Measurement scheme
        rhomap: Resistivity distribution
        noise_levels: List of noise percentages
        
    Returns:
        datasets: List of synthetic datasets
    """
    datasets = []
    
    for i, noise in enumerate(noise_levels):
        print(f"Generating dataset with {noise*100:.0f}% noise...")
        
        # Simulate ERT data with specified noise level
        data = ert.simulate(mesh, scheme=scheme, res=rhomap, 
                           noiseLevel=noise, noiseAbs=1e-6, seed=1337+i)
        
        # Quality control checks
        negative_rhoa = np.sum(data['rhoa'] < 0)
        print(f"  Negative values: {negative_rhoa}")
        print(f"  Rhoa range: {np.min(data['rhoa']):.1f} - {np.max(data['rhoa']):.1f} Ωm")
        print(f"  Error range: {np.min(data['err'])*100:.1f}% - {np.max(data['err'])*100:.1f}%")
        print(f"  Number of measurements: {data.size()}")
        
        # Save dataset
        data.save(f'synthetic_data_{noise*100:.0f}percent_noise.dat')
        datasets.append(data)
        print(f" Saved as 'synthetic_data_{noise*100:.0f}percent_noise.dat'")
    
    return datasets


def plot_true_model_and_pseudosections(mesh, rhomap, datasets, noise_levels):
    """
    Plot true model and pseudosections for all noise levels.
    
    Args:
        mesh: Computational mesh with true model
        rhomap: Resistivity distribution
        datasets: List of synthetic datasets
        noise_levels: List of noise percentages
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. True model (top left)
    pg.show(mesh, data=rhomap, label=pg.unit('res'), ax=axes[0], 
            showMesh=True, cMap="Spectral_r", cMin=30, cMax=500)
    axes[0].set_title('True Model', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_xlim(-40, 40)
    
    # 2. Pseudosections for each noise level
    for i, (noise, data) in enumerate(zip(noise_levels, datasets), 1):
        ert.show(data, ax=axes[i], cMap="rainbow", cMin=30, cMax=150, showElectrodes=False)
        axes[i].set_title(f'Pseudosection - {noise*100:.0f}% Noise', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Distance (m)')
        axes[i].set_ylabel('Depth (m)')
        axes[i].set_xlim(-20, 20)
    
    plt.tight_layout()
    plt.savefig('inversion_results/true_model_and_pseudosections.png', dpi=300, bbox_inches='tight')
    plt.close()


def setup_inversion_algorithms():
    """
    Define inversion algorithms and their parameters.
    
    Returns:
        algorithms: Dictionary of algorithm parameters
    """
    algorithms = {
        "Occam": {"lam": 100, "robust": False},      # Smooth inversion
        "Marquardt": {"lam": 30, "robust": False},   # Sensitive inversion
        "Robust": {"lam": 50, "robust": True}        # Robust inversion
    }
    return algorithms


def run_inversions(datasets, noise_levels, algorithms):
    """
    Run ERT inversions for all algorithm and noise level combinations.
    
    Args:
        datasets: List of synthetic datasets
        noise_levels: List of noise percentages
        algorithms: Dictionary of algorithm parameters
        
    Returns:
        results: Dictionary containing inversion results
    """
    results = {algo_name: {} for algo_name in algorithms.keys()}
    
    # Create output directories
    for algo_name in algorithms.keys():
        os.makedirs(f"inversion_results/inversion_{algo_name}/plots", exist_ok=True)
    
    for algo_name, params in algorithms.items():
        print(f"INVERSION WITH {algo_name.upper()} ALGORITHM")
        
        for i, (noise, data) in enumerate(zip(noise_levels, datasets)):
            print(f"\n{algo_name} inversion for {noise*100:.0f}% noise...")
            
            # Initialize ERT manager
            mgr = ert.ERTManager(data)
            
            # Enable robust inversion if specified
            if params.get("robust", False):
                mgr.inv.robustData = True
            
            # Run inversion
            inv = mgr.invert(lam=params["lam"], verbose=False)
            rms = np.sqrt(mgr.inv.chi2())
            
            # Store results
            results[algo_name][noise] = {
                "model": mgr.model,
                "predicted_data": mgr.inv.response,
                "observed_data": mgr.data,
                "rms": rms
            }
            
            # Create comparison figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pseudosection
            ert.show(data, ax=ax1, cMap="rainbow", cMin=30, cMax=150, showElectrodes=False)
            ax1.set_title('Apparent Resistivity Pseudosection', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Distance (m)', fontsize=12)
            ax1.set_ylabel('Depth (m)', fontsize=12)
            ax1.set_xlim(-20, 20)
            
            # Inversion result
            mgr.showResult(ax=ax2, cMin=30, cMax=150, showElectrodes=False, cMap="rainbow")
            ax2.set_title(f'{algo_name} Inversion (RMS: {rms:.3f})', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Distance (m)', fontsize=12)
            ax2.set_ylabel('Depth (m)', fontsize=12)
            ax2.set_xlim(-20, 20)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f'inversion_results/inversion_{algo_name}/plots/{algo_name}_{noise*100:.0f}percent_noise.png', 
                   dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f" RMS: {rms:.3f} - Saved to inversion_{algo_name}/")
    
    return results


def print_rms_table(results, noise_levels):
    """
    Print RMS values in a formatted table.
    
    Args:
        results: Dictionary containing inversion results
        noise_levels: List of noise percentages
    """
    print("\n ~FINAL RMS TABLE~")
    
    # Table header
    header = ["Algorithm"] + [f"{int(n*100)}%" for n in noise_levels]
    print("{:<12}".format("Algorithm"), end="")
    for h in header[1:]:
        print("{:>12}".format(h), end="")
    print()
    
    # Table rows
    for algo_name in results:
        print("{:<12}".format(algo_name), end="")
        for n in noise_levels:
            val = results[algo_name][n]["rms"]
            print("{:>12.3f}".format(val), end="")
        print()


def plot_rms_vs_noise(results, noise_levels):
    """
    Plot RMS values vs noise level for all algorithms.
    
    Args:
        results: Dictionary containing inversion results
        noise_levels: List of noise percentages
    """
    plt.figure(figsize=(8, 5))
    
    for algo_name in results:
        rms_values = [results[algo_name][n]["rms"] for n in noise_levels]
        plt.plot([n*100 for n in noise_levels], rms_values, marker='o', label=algo_name)
    
    plt.xlabel("Noise level (%)")
    plt.ylabel("RMS")
    plt.title("RMS vs Noise level for different inversion algorithms")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('inversion_results/rms_vs_noise.png', dpi=300, bbox_inches='tight')
    plt.close()


def perform_residual_analysis(results, algorithms, noise_levels, datasets):
    """
    Perform comprehensive residual analysis for all inversion results.
    
    Args:
        results: Dictionary containing inversion results
        algorithms: Dictionary of algorithm parameters
        noise_levels: List of noise percentages
        datasets: List of synthetic datasets
    """
    for algo_name in algorithms.keys():
        print(f"RESIDUAL ANALYSIS FOR {algo_name.upper()}")
        
        
        for i, (noise, data) in enumerate(zip(noise_levels, datasets)):
            result = results[algo_name][noise]
            
            # Calculate residuals
            observed = result["observed_data"]["rhoa"]
            predicted = result["predicted_data"]
            residuals = observed - predicted
            
            # Calculate normalized residuals
            errors = result["observed_data"]["err"] * observed
            normalized_residuals = residuals / errors
            
            # Store residuals
            results[algo_name][noise]["residuals"] = residuals
            results[algo_name][noise]["normalized_residuals"] = normalized_residuals
            
            # Print statistics
            print(f"\nNoise {noise*100:.0f}%:")
            print(f"  RMS: {result['rms']:.3f}")
            print(f"  Mean absolute residual: {np.mean(np.abs(residuals)):.3f} Ωm")
            print(f"  Std residuals: {np.std(residuals):.3f} Ωm")
            print(f"  Mean normalized residual: {np.mean(np.abs(normalized_residuals)):.3f}")
            print(f"  Percentage |normalized residuals| > 3: "
                  f"{np.sum(np.abs(normalized_residuals) > 3) / len(normalized_residuals) * 100:.1f}%")
    
    return results


def plot_normalized_residuals_histograms(results, algorithms, noise_levels, datasets):
    """
    Plot histograms of normalized residuals for all algorithm and noise combinations.
    
    Args:
        results: Dictionary containing inversion results
        algorithms: Dictionary of algorithm parameters
        noise_levels: List of noise percentages
        datasets: List of synthetic datasets
    """
    for algo_name in algorithms.keys():
        # Create residuals subdirectory
        residual_dir = f"inversion_results/inversion_{algo_name}/residuals"
        os.makedirs(residual_dir, exist_ok=True)
        
        for i, (noise, data) in enumerate(zip(noise_levels, datasets)):
            result = results[algo_name][noise]
            norm_residuals = result["normalized_residuals"]
            
            # Create histogram figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot histogram of normalized residuals
            n, bins, patches = ax.hist(norm_residuals, bins=30, alpha=0.7, 
                                      edgecolor='black', density=True, color='skyblue')
            
            # Plot Gaussian distribution for comparison
            x = np.linspace(-4, 4, 100)
            ax.plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=2, label='Gaussian Distribution')
            
            ax.set_xlabel('Normalized Residuals', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title(f'Normalized Residuals Distribution - {algo_name} ({noise*100:.0f}% Noise)\nRMS: {result["rms"]:.3f}', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = (f'Mean: {np.mean(norm_residuals):.3f}\n'
                         f'Std: {np.std(norm_residuals):.3f}')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            # Save figure
            filename = f"{residual_dir}/{algo_name}_{noise*100:.0f}percent_normalized_residuals.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    print("All normalized residual histograms saved successfully!")


def main():
    """Main function to run the complete ERT inversion and analysis workflow."""
    
    # Create main results directory
    os.makedirs("inversion_results", exist_ok=True)
    
    # Configuration
    noise_levels = [0.01, 0.02, 0.05, 0.10, 0.20]  # 1%, 2%, 5%, 10%, 20%
    
    # Create geometry and mesh
    print("Creating geological geometry...")
    geom = create_geometry()
    
    print("Setting up measurement scheme...")
    scheme = setup_measurement_scheme(geom)
    
    print("Creating mesh...")
    mesh = mt.createMesh(geom, quality=24)
    
    # Define resistivity distribution
    print("Defining resistivity map...")
    rhomap = create_resistivity_map()
    
    # Generate synthetic datasets
    print("Generating synthetic datasets...")
    datasets = generate_synthetic_datasets(mesh, scheme, rhomap, noise_levels)
    
    # Plot true model and pseudosections
    print("Plotting true model and pseudosections...")
    plot_true_model_and_pseudosections(mesh, rhomap, datasets, noise_levels)
    
    # Setup inversion algorithms
    print("Setting up inversion algorithms...")
    algorithms = setup_inversion_algorithms()
    
    # Run inversions
    print("Running inversions...")
    results = run_inversions(datasets, noise_levels, algorithms)
    
    # Print and plot RMS results
    print_rms_table(results, noise_levels)
    plot_rms_vs_noise(results, noise_levels)
    
    # Perform residual analysis
    print("Performing residual analysis...")
    results = perform_residual_analysis(results, algorithms, noise_levels, datasets)
    
    # Plot normalized residuals
    print("Plotting normalized residuals histograms...")
    plot_normalized_residuals_histograms(results, algorithms, noise_levels, datasets)
    
    print("\n Analysis completed successfully!")


if __name__ == "__main__":
    main()