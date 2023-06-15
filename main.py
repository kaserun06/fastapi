import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from typing import List
import shutil
# from google.cloud import storage
# import firebase_admin
# from firebase_admin import credentials, firestore
from datetime import datetime
from pydantic import BaseModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = FastAPI()  # create a new FastAPI app instance

# port = int(os.getenv("PORT"))
port = 8080

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

data_source = pd.read_csv("PATH KE DATA SOURCE")

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: 5

def preprocess_data(data):
    data["hashed_username"] = tf.strings.to_hash_bucket_fast(data["username"], 1000)
    data["hashed_id"] = tf.strings.to_hash_bucket_fast(data["id_griya"], 1000)
    return data

def create_dataset(data):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {"id_user": data["hashed_username"], "id_griya": data["hashed_id"]},
            data["reviews.rating"],
        )
    )
    return dataset

def create_hotel_name_mapping(data_preprocessed):
    hotel_name_mapping = {}
    for _, row in data_preprocessed.iterrows():
        hotel_name_mapping[row["hashed_id"]] = row["name"]
    return hotel_name_mapping

    

def get_recommendations(user_id, model, num_recommendations, hotel_name_mapping):
    user_embedding = model.user_embedding(np.array([user_id]))
    hotel_embeddings = model.hotel_embedding(np.arange(len(hotel_name_mapping)))
    similarities = np.dot(user_embedding, hotel_embeddings.numpy().T).flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    top_hotel_indices = sorted_indices[:num_recommendations]
    recommended_hotels = [hotel_name_mapping[index] for index in top_hotel_indices]
    return recommended_hotels


@app.post("/recommend")
def recommend_hotels(request: RecommendationRequest) -> List[str]:
    user_id = request.user_id
    num_recommendations = request.num_recommendations

    # Load your model
    model = tf.keras.models.load_model('hotel_recommendation_model')

    data = data_source
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    preprocessed_data = preprocess_data(data)
    hotel_name_mapping = create_hotel_name_mapping(preprocessed_data)
    train_dataset = create_dataset(train_data)

    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.fit(train_dataset, epochs=5)

    # Call the get_recommendations function here to get the recommended griya
    recommended_hotels = get_recommendations(
        user_id, model, num_recommendations, hotel_name_mapping
    )

    return recommended_hotels



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
