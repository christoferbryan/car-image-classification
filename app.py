import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Car Model Classifier",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Stanford Cars Classifier")
st.markdown("""
Upload an image of a car, and the model will predict its make, model, and year!
This model is trained on the Stanford Cars dataset with 196 different car classes.
""")

# Device configuration
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Load class names
@st.cache_data
def load_class_names():
    """Load class names from the Stanford Cars dataset"""
    try:
        from datasets import load_dataset
        dataset = load_dataset("tanganke/stanford_cars")
        class_names = dataset['train'].features['label'].names
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        # Fallback to generic names if dataset can't be loaded
        return [f"Class {i}" for i in range(196)]

class_names = load_class_names()

# Define the model architecture
def create_model(num_classes=196):
    """Create the ResNet18 model with custom FC layers"""
    model = models.resnet18(pretrained=False)
    
    # Replace FC layer with the same architecture used in training
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_classes)
    )
    
    return model

# Load model
@st.cache_resource
def load_model():
    """Load the trained model"""
    model = create_model(num_classes=len(class_names))
    
    try:
        # Load the trained weights
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.to(device)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'best_model.pth' not found. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Define image preprocessing
def get_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

transform = get_transforms()

# Prediction function
def predict_image(image, model, transform, class_names, device, top_k=5):
    """Predict the class of an uploaded image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Get top predictions
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'class': class_names[idx],
            'probability': float(prob),
            'index': int(idx)
        })
    
    return predictions

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model Information:**
    - Architecture: ResNet18 (Transfer Learning)
    - Dataset: Stanford Cars (196 classes)
    - Training: Two-phase (Feature Extraction + Fine-tuning)
    
    **How to use:**
    1. Upload a car image (JPG, PNG, JPEG)
    2. View the prediction results
    3. See top-5 most likely car models
    """)
    
    st.header("⚙️ Settings")
    show_top_k = st.slider("Number of predictions to show", 1, 10, 5)
    confidence_threshold = st.slider("Confidence threshold (%)", 0, 100, 10)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a car image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a car (front, side, or angled view works best)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')

with col2:
    st.header("🎯 Prediction Results")
    
    if uploaded_file is not None and model is not None:
        with st.spinner('🔍 Analyzing image...'):
            try:
                # Make prediction
                predictions = predict_image(
                    image, model, transform, class_names, device, top_k=show_top_k
                )
                
                # Display top prediction prominently
                top_pred = predictions[0]
                st.success(f"**Predicted Car:** {top_pred['class']}")
                st.metric(
                    label="Confidence", 
                    value=f"{top_pred['probability']*100:.2f}%"
                )
                
                # Display all top-k predictions
                st.subheader(f"Top {show_top_k} Predictions")
                
                for i, pred in enumerate(predictions):
                    confidence = pred['probability'] * 100
                    
                    # Only show if above threshold
                    if confidence >= confidence_threshold:
                        # Create a colored bar based on confidence
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.write(f"**{i+1}. {pred['class']}**")
                        
                        with col_b:
                            st.write(f"{confidence:.2f}%")
                        
                        # Progress bar
                        st.progress(pred['probability'])
                        
                        st.divider()
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
    
    elif uploaded_file is None:
        st.info("Upload an image to get started!")
    
    else:
        st.warning("⚠️ Model not loaded. Please check the model file.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Model trained on Stanford Cars Dataset</p>
</div>
""", unsafe_allow_html=True)

# Example images section (optional)
# with st.expander("💡 Tips for better predictions"):
#     st.markdown("""
#     - Use clear, well-lit images
#     - Car should be the main subject
#     - Front, side, or 3/4 angle views work best
#     - Avoid heavily cropped or blurry images
#     - Higher resolution images generally work better
#     """)

# Display model info in expander
if model is not None:
    with st.expander("🔧 Model Architecture Details"):
        st.markdown(f"""
Model: ResNet18 with Custom Classifier

Backbone: ResNet18 (Pretrained on ImageNet)
Classifier Architecture:
  - Linear(512 → 512)
  - BatchNorm1d(512)
  - ReLU
  - Dropout(0.3)
  - Linear(512 → 256)
  - BatchNorm1d(256)
  - ReLU
  - Dropout(0.2)
  - Linear(256 → 196)

Total Parameters: ~11.7M \n
Trainable FC Parameters: ~465K \n
Number of Classes: {len(class_names)} \n
Device: {device}
        """)