"""
Create scatter plots for SAE feature analysis.

This module contains a function to create scatter plots of feature distributions.
"""

import numpy as np
from scipy import stats
import plotly.graph_objs as go


def plot_features_scatter(layer_features, k_features=10, labeled_features=None, 
                         title=None, top_percentile=0.1, show_labels=False,
                         width=800, height=600):
    """
    Plot features on a scatter plot with logarithmic scales using Plotly.
    
    Args:
        layer_features (LayerFeatures): LayerFeatures object with topk_filtered and bottomk_filtered methods
        k_features (int): Number of top features to highlight for each category (target, benign, shared)
        labeled_features (list, optional): List of Feature objects or feature indices to label with feature numbers
        title (str): Plot title
        top_percentile (float): Fraction of most frequent features to plot (e.g., 0.1 for top 10%)
        show_labels (bool): Whether to show feature indices as text labels
        width (int): Figure width in pixels
        height (int): Figure height in pixels
    
    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    circle_line_width = 2
    
    # Get target and benign features using the LayerFeatures methods
    top_target_features = layer_features.topk_filtered(k_features)
    top_benign_features = layer_features.bottomk_filtered(k_features)
    
    # Get shared features (high frequency but not top target or benign)
    feature_frequencies = [(f.target_count + f.benign_count, f) for f in layer_features]
    feature_frequencies.sort(key=lambda x: x[0], reverse=True)
    
    # Filter out features that are already in target or benign
    target_indices = {f.index for f in top_target_features}
    benign_indices = {f.index for f in top_benign_features}
    
    top_shared_features = []
    for freq, f in feature_frequencies:
        if f.index not in target_indices and f.index not in benign_indices:
            top_shared_features.append(f)
        if len(top_shared_features) >= k_features:
            break
    
    # Calculate frequency as sum of target_count and benign_count for plotting all features
    feature_frequencies = [(f.target_count + f.benign_count, f) for f in layer_features]
    
    # Sort by frequency and take top percentile
    feature_frequencies.sort(key=lambda x: x[0], reverse=True)
    num_top_features = int(len(feature_frequencies) * top_percentile)
    top_features = [f for freq, f in feature_frequencies[:num_top_features]]
    
    print(f"Plotting top {top_percentile*100:.1f}% most frequent features: {len(top_features)} out of {len(layer_features)}")
    print(f"Target features: {[f.index for f in top_target_features]}")
    print(f"Benign features: {[f.index for f in top_benign_features]}")
    print(f"Shared features: {[f.index for f in top_shared_features]}")
    
    # Extract data from top features
    x_values = []
    y_values = []
    colors = []
    labels = []
    feature_indices = []
    
    for f in top_features:
        if f.benign_count > 0 and f.target_count > 0 and f.index != 15530:  # Filter out zero values for log scale
            x_values.append(f.benign_count)
            y_values.append(f.target_count)
            # Use log of target_acts_relative for coloring, handle zero/negative values
            log_relative = np.log10(f.target_acts_relative) if f.target_acts_relative > 0 else -10
            colors.append(log_relative)
            labels.append(f"Feature {f.index}<br>Benign: {f.benign_count}<br>Target: {f.target_count}<br>Relative: {f.target_acts_relative:.4f}")
            feature_indices.append(str(f.index))
    
    if not x_values:
        print("No valid data points found (all have zero counts)")
        return None
    
    # Calculate correlation and p-value
    correlation, p_value = stats.pearsonr(np.log10(x_values), np.log10(y_values))

    # Create Plotly figure
    fig = go.Figure()
    
    # Set the marker mode based on show_labels parameter
    marker_mode = 'markers+text' if show_labels else 'markers'
    
    # Configure marker properties
    marker_props = {
        'size': 12,
        'opacity': 0.7,
        'line': dict(width=1, color='black'),
        'color': colors,
        'colorscale': 'Viridis',
        'colorbar': dict(
            title=dict(
                text='Log Target Acts Relative',
                side='right'
            ),
            thickness=15,
            len=0.7,
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
        )
    }
    
    # Add main scatter plot
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode=marker_mode,
        marker=marker_props,
        text=feature_indices if show_labels else None,
        textposition="top center",
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=labels,
        name='Features'
    ))
    
    # Highlight target features with red circles
    if top_target_features:
        target_features_indices = {f.index for f in top_target_features}
        target_x = []
        target_y = []
        target_labels = []
        
        for i, f in enumerate(top_features):
            if f.index in target_features_indices and f.benign_count > 0 and f.target_count > 0:
                target_x.append(f.benign_count)
                target_y.append(f.target_count)
                target_labels.append(f"TOP TARGET Feature {f.index}<br>Benign: {f.benign_count}<br>Target: {f.target_count}<br>Relative: {f.target_acts_relative:.4f}")
        
        if target_x:
            fig.add_trace(go.Scatter(
                x=target_x,
                y=target_y,
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='circle-open',
                    color='#CF4A57',
                    line=dict(width=circle_line_width, color='#CF4A57'),
                    opacity=1.0
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=target_labels,
                name='Top Target Features',
                showlegend=False
            ))
    
    # Highlight benign features with blue circles
    if top_benign_features:
        benign_features_indices = {f.index for f in top_benign_features}
        benign_x = []
        benign_y = []
        benign_labels = []
        
        for i, f in enumerate(top_features):
            if f.index in benign_features_indices and f.benign_count > 0 and f.target_count > 0:
                benign_x.append(f.benign_count)
                benign_y.append(f.target_count)
                benign_labels.append(f"TOP BENIGN Feature {f.index}<br>Benign: {f.benign_count}<br>Target: {f.target_count}<br>Relative: {f.target_acts_relative:.4f}")
        
        if benign_x:
            fig.add_trace(go.Scatter(
                x=benign_x,
                y=benign_y,
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='circle-open',
                    color='blue',
                    line=dict(width=circle_line_width, color='blue'),
                    opacity=1.0
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=benign_labels,
                name='Top Benign Features',
                showlegend=False
            ))
    
    # Highlight shared features with magenta circles
    if top_shared_features:
        shared_features_indices = {f.index for f in top_shared_features}
        shared_x = []
        shared_y = []
        shared_labels = []
        
        for i, f in enumerate(top_features):
            if f.index in shared_features_indices and f.benign_count > 0 and f.target_count > 0:
                shared_x.append(f.benign_count)
                shared_y.append(f.target_count)
                shared_labels.append(f"TOP SHARED Feature {f.index}<br>Benign: {f.benign_count}<br>Target: {f.target_count}<br>Relative: {f.target_acts_relative:.4f}")
        
        if shared_x:
            fig.add_trace(go.Scatter(
                x=shared_x,
                y=shared_y,
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='circle-open',
                    color='#9932CC',
                    line=dict(width=circle_line_width, color='#9932CC'),
                    opacity=1.0
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=shared_labels,
                name='Top Shared Features',
                showlegend=False
            ))
    
    # Add feature number labels for specific features
    if labeled_features:
        # Convert labeled_features to indices if they are Feature objects
        if hasattr(labeled_features[0], 'index'):
            labeled_indices = {f.index for f in labeled_features}
        else:
            labeled_indices = set(labeled_features)
        
        labeled_x = []
        labeled_y = []
        labeled_labels = []
        
        for f in top_features:
            if f.index in labeled_indices and f.benign_count > 0 and f.target_count > 0:
                labeled_x.append(f.benign_count)
                labeled_y.append(f.target_count)
                labeled_labels.append(f"LABELED Feature {f.index}<br>Benign: {f.benign_count}<br>Target: {f.target_count}<br>Relative: {f.target_acts_relative:.4f}")
        
        if labeled_x:
            fig.add_trace(go.Scatter(
                x=labeled_x,
                y=labeled_y,
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='circle-open',
                    color='red',
                    line=dict(width=3, color='red'),
                    opacity=1.0
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=labeled_labels,
                name='Labeled Features',
                showlegend=False
            ))
    
    # Set logarithmic scales and layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top'
        ) if title else None,
        xaxis=dict(
            type="log",
            title=dict(text="Benign Count", font=dict(size=20)),
            tickfont=dict(size=16),
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            showticklabels=True,
        ),
        yaxis=dict(
            type="log",
            title=dict(text="Target Count", font=dict(size=20)),
            tickfont=dict(size=16),
            gridcolor='lightgray',
            gridwidth=1,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            showticklabels=True,
        ),
        template="plotly_white",
        showlegend=False,
        width=width,
        height=height,
        font=dict(size=16),
        margin=dict(l=80, r=20, t=40, b=80),
    )

    # Show the plot
    fig.show()
