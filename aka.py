import streamlit as st  
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import random
import math

# Product class to represent e-commerce items
SELECTION_SORT = 'Selection Sort'
MERGE_SORT = 'Merge Sort'

class Product:
    def __init__(self, id: int, name: str, price: float, rating: float, 
                 stock: int, category: str, date_added: datetime):
        self.id = id
        self.name = name
        self.price = price
        self.rating = rating
        self.stock = stock
        self.category = category
        self.date_added = date_added

    def to_dict(self) -> Dict[str, Any]:
        field_mapping = {
            'ID': 'id',
            'Name': 'name',
            'Price': 'price',
            'Rating': 'rating',
            'Stock': 'stock',
            'Category': 'category',
            'Date Added': 'date_added'
        }
        return {k: getattr(self, field_mapping[k]) for k in field_mapping.keys()}

# Class for sorting algorithms
class SortingAnalyzer:
    @staticmethod
    def selection_sort(products: List[Product], key: str, ascending: bool = True) -> Tuple[List[Product], Dict[str, Any]]:
        comparisons = swaps = 0
        steps = []
        start_time = time.time()
        n = len(products)
        
        # Calculate theoretical complexity for Selection Sort
        theoretical_complexity = (n * n - n) / 2
        
        def compare_products(a: Product, b: Product) -> bool:
            nonlocal comparisons
            comparisons += 1
            a_val = getattr(a, key.lower())
            b_val = getattr(b, key.lower())
            return a_val < b_val if ascending else a_val > b_val
        
        # Main sorting loop
        for i in range(n):
            min_idx = i
            # Find minimum element
            for j in range(i + 1, n):
                if compare_products(products[j], products[min_idx]):
                    min_idx = j
            
            # Swap if needed and record step
            if min_idx != i:
                products[i], products[min_idx] = products[min_idx], products[i]
                swaps += 1
                steps.append({
                    'step': i + 1,
                    'comparisons': comparisons,
                    'swaps': swaps
                })
        
        return products, {
            'execution_time': time.time() - start_time,
            'comparisons': comparisons,
            'swaps': swaps,
            'total_operations': comparisons + swaps,
            'theoretical_complexity': theoretical_complexity,
            'steps': steps
        }
    @staticmethod
    def merge_sort(products: List[Product], key: str, ascending: bool = True) -> Tuple[List[Product], Dict[str, Any]]:
        comparisons = 0
        merges = 0
        steps = []
        start_time = time.time()
        n = len(products)
        
        # Calculate theoretical complexity for Merge Sort
        theoretical_complexity = n * math.log2(n) if n > 0 else 0

        def merge(left: List[Product], right: List[Product]) -> List[Product]:
            nonlocal comparisons, merges
            result = []
            i = j = 0

            while i < len(left) and j < len(right):
                comparisons += 1
                left_val = getattr(left[i], key.lower())
                right_val = getattr(right[j], key.lower())

                if ascending:
                    should_take_left = left_val <= right_val
                else:
                    should_take_left = left_val >= right_val

                if should_take_left:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1

            result.extend(left[i:])
            result.extend(right[j:])
            merges += 1
            steps.append({
                'step': len(steps) + 1,
                'comparisons': comparisons,
                'merges': merges
            })
            return result

        def sort(arr: List[Product]) -> List[Product]:
            if len(arr) <= 1:
                return arr

            mid = len(arr) // 2
            left = sort(arr[:mid])
            right = sort(arr[mid:])

            return merge(left, right)

        sorted_products = sort(products.copy())
        execution_time = time.time() - start_time

        return sorted_products, {
            'execution_time': execution_time,
            'comparisons': comparisons,
            'merges': merges,
            'total_operations': comparisons + merges,
            'theoretical_complexity': theoretical_complexity,
            'steps': steps
        }

class DataGenerator:
    CATEGORIES = ['Electronics', 'Clothing', 'Book', 'Home & Garden', 'Sports', 'Toys']
    PRODUCT_NAMES = ['Laptop', 'Phone', 'Tablet', 'Watch', 'Camera', 'Headphones']
    
    @staticmethod
    def generate_sample_data(n: int) -> List[Product]:
        products = []
        for i in range(n):
            name = f"{random.choice(['Premium', 'Basic', 'Pro'])} {random.choice(DataGenerator.PRODUCT_NAMES)}"
            products.append(Product(
                id=i+1,
                name=name,
                price=round(random.uniform(10, 1000), 2),
                rating=round(random.uniform(1, 5), 1),
                stock=random.randint(0, 100),
                category=random.choice(DataGenerator.CATEGORIES),
                date_added=datetime.now() - timedelta(days=random.randint(0, 365))
            ))
        return products

def create_performance_chart(selection_metrics: Dict[str, Any], merge_metrics: Dict[str, Any], n_products: int) -> go.Figure:
    metrics_df = pd.DataFrame({
        'Metric': ['Time (seconds)', 'Total Operations', 'Theoretical Complexity'],
        SELECTION_SORT: [
            selection_metrics['execution_time'],
            selection_metrics['total_operations'],
            selection_metrics['theoretical_complexity']
        ],
        MERGE_SORT: [
            merge_metrics['execution_time'],
            merge_metrics['total_operations'],
            merge_metrics['theoretical_complexity']
        ]
    })

    fig = go.Figure()
    for metric in metrics_df['Metric']:
        fig.add_trace(go.Bar(
            name=metric,
            x=[SELECTION_SORT, MERGE_SORT],
            y=metrics_df[metrics_df['Metric'] == metric].iloc[0, 1:],
            text=metrics_df[metrics_df['Metric'] == metric].iloc[0, 1:].round(6),
            textposition='auto',
        ))

    fig.update_layout(
        title=f'Performance Comparison (n={n_products})',
        barmode='group',
        yaxis_type="log",
        height=500
    )
    return fig

def create_runtime_comparison_chart(selection_metrics: Dict[str, Any], merge_metrics: Dict[str, Any], n_products: int) -> go.Figure:
    fig = go.Figure()
    
    # Generate points for x-axis
    x_points = list(range(1, n_products + 1))
    
    # Calculate theoretical points
    selection_theoretical = [(n * n - n) / 2 for n in x_points]
    merge_theoretical = [n * math.log2(n) if n > 0 else 0 for n in x_points]
    
    # Merge Sort actual and theoretical lines
    fig.add_trace(go.Scatter(
        x=x_points,
        y=merge_theoretical,
        mode='lines',
        name='Merge Sort (Theoretical)',
        line=dict(color='blue', dash='dash')
    ))
    
    # Selection Sort actual and theoretical lines
    fig.add_trace(go.Scatter(
        x=x_points,
        y=selection_theoretical,
        mode='lines',
        name='Selection Sort (Theoretical)',
        line=dict(color='red', dash='dash')
    ))
    
    # Add current points
    fig.add_trace(go.Scatter(
        x=[n_products],
        y=[merge_metrics['total_operations']],
        mode='markers',
        name='Merge Sort (Actual)',
        marker=dict(color='blue', size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=[n_products],
        y=[selection_metrics['total_operations']],
        mode='markers',
        name='Selection Sort (Actual)',
        marker=dict(color='red', size=10)
    ))

    # Calculate y-axis range to ensure proper scaling
    max_y = max(
        max(selection_theoretical),
        max(merge_theoretical),
        selection_metrics['total_operations'],
        merge_metrics['total_operations']
    )
    min_y = min(
        min(y for y in selection_theoretical if y > 0),
        min(y for y in merge_theoretical if y > 0),
        selection_metrics['total_operations'],
        merge_metrics['total_operations']
    )

    # Set y-axis range with some padding
    y_range = [min_y * 0.8, max_y * 1.2]

    fig.update_layout(
        title='Runtime Comparison: Actual vs Theoretical',
        xaxis_title='Input Size (n)',
        yaxis_title='Number of Operations',
        yaxis_type="log",
        yaxis=dict(
            range=[math.log10(y_range[0]), math.log10(y_range[1])],
            dtick=1  # Set tick interval to 1 log unit
        ),
        height=500,
        showlegend=True
    )
    return fig

def create_complexity_classes_chart(n_products: int) -> go.Figure:
    # Generate data points based on actual input size
    n = np.arange(1, n_products + 1)
    
    # Calculate complexity values
    quadratic = n**2  # O(n²)
    loglinear = n * np.log2(n)  # O(n log n)
    
    # Calculate y-axis range based on theoretical values only
    max_y = max(max(quadratic), max(loglinear))
    
    fig = go.Figure()
    
    # Add O(n log n) line
    fig.add_trace(go.Scatter(
        x=n,
        y=loglinear,
        mode='lines',
        name='Merge Sort ∈ O(n log n)',
        line=dict(color='orange'),
    ))
    
    # Add O(n²) line
    fig.add_trace(go.Scatter(
        x=n,
        y=quadratic,
        mode='lines',
        name='Selection Sort ∈ O(n²)',
        line=dict(color='red'),
    ))

    fig.update_layout(
        title='Complexity Classes Comparison',
        xaxis_title='Input Size (n)',
        yaxis_title='Running Time',
        height=500,
        showlegend=True,
        yaxis=dict(
            range=[0, max_y * 1.1],  # Add 10% padding to y-axis
        ),
        xaxis=dict(
            range=[0, n_products * 1.1]  # Add 10% padding to x-axis
        )
    )
    return fig

def main():
    st.set_page_config(page_title="E-commerce Sorting Analysis", layout="wide")

    st.title("🛍️ E-commerce Sorting Algorithm Analysis")
    st.write("Compare the performance of Selection Sort and Merge Sort algorithms")

    # Initialize session state for storing generated data
    if 'products' not in st.session_state:
        st.session_state.products = None
        st.session_state.data_generated = False

    # Sidebar configuration
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.header("⚙️ Settings")

        manual_input = st.number_input(
            "Manual Input for Number of Products",
            min_value=1,
            max_value=10000,
            value=100,
            step=1,
            key="manual_input",
        )

        n_products = st.slider(
            "Number of Products (Slider)",
            min_value=1,
            max_value=10000,
            value=manual_input if manual_input else 100,
            key="n_products_slider",
        )

        if manual_input != n_products:
            n_products = manual_input

        # Sort settings
        sort_key = st.selectbox(
            "Sort by", ["ID", "Name", "Price", "Rating", "Stock"], key="sort_key"
        )
        sort_order = st.selectbox("Sort Order", ["Ascending", "Descending"], key="sort_order")

        # Generate Data button
        generate_data = st.button("Generate Data & Run Analysis", key="generate_data")

        st.markdown("</div>", unsafe_allow_html=True)

    # Generate new data when button is clicked
    if generate_data:
        with st.spinner("Generating new data..."):
            st.session_state.products = DataGenerator.generate_sample_data(n_products)
            st.session_state.data_generated = True

    # Process and display data if it exists
    if st.session_state.data_generated and st.session_state.products:
        with st.spinner("Processing data..."):
            # Perform sorting
            ascending = sort_order == "Ascending"
            selection_sorted, selection_metrics = SortingAnalyzer.selection_sort(
                st.session_state.products.copy(), sort_key.lower(), ascending
            )
            merge_sorted, merge_metrics = SortingAnalyzer.merge_sort(
                st.session_state.products.copy(), sort_key.lower(), ascending
            )

            # Create DataFrames
            original_df = pd.DataFrame([p.to_dict() for p in st.session_state.products])
            selection_df = pd.DataFrame([p.to_dict() for p in selection_sorted])
            merge_df = pd.DataFrame([p.to_dict() for p in merge_sorted])

            # Reset index to start from 1
            original_df.index = range(1, len(original_df) + 1)
            selection_df.index = range(1, len(selection_df) + 1)
            merge_df.index = range(1, len(merge_df) + 1)

            # Display results in tabs
            tabs = st.tabs(["📊 Data View", "📈 Performance Analysis", "📚 Theoretical Analysis", "📉 Runtime Graphs"])

            with tabs[0]:
                st.subheader("Original Data")
                st.dataframe(original_df)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Selection Sort Results")
                    st.dataframe(selection_df)
                with col2:
                    st.subheader("Merge Sort Results")
                    st.dataframe(merge_df)

            with tabs[1]:
                st.plotly_chart(
                    create_performance_chart(selection_metrics, merge_metrics, n_products),
                    use_container_width=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Selection Sort Metrics")
                    st.json(selection_metrics)
                with col2:
                    st.subheader("Merge Sort Metrics")
                    st.json(merge_metrics)

            with tabs[2]:
                st.subheader("Theoretical vs Actual Performance")
                
                selection_ratio = (
                    selection_metrics['total_operations'] / selection_metrics['theoretical_complexity']
                    if selection_metrics['theoretical_complexity'] > 0 else None
                )
                merge_ratio = (
                    merge_metrics['total_operations'] / merge_metrics['theoretical_complexity']
                    if merge_metrics['theoretical_complexity'] > 0 else None
                )
                
                theory_df = pd.DataFrame({
                    'Algorithm': ['Selection Sort', 'Merge Sort'],
                    'Theoretical Complexity': [
                        selection_metrics['theoretical_complexity'],
                        merge_metrics['theoretical_complexity'],
                    ],
                    'Actual Operations': [
                        selection_metrics['total_operations'],
                        merge_metrics['total_operations'],
                    ],
                    'Ratio (Actual/Theoretical)': [
                        selection_ratio,
                        merge_ratio,
                    ],
                })
                st.dataframe(theory_df)

            with tabs[3]:
                st.plotly_chart(
                    create_runtime_comparison_chart(selection_metrics, merge_metrics, n_products),
                    use_container_width=True,
                )
                st.plotly_chart(
                    create_complexity_classes_chart(n_products),
                    use_container_width=True
                )
    else:
        st.info("👆 Click 'Generate New Data & Run Analysis' button in the sidebar to start the analysis")

if __name__ == "__main__":
    main()
