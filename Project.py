import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
# image_path="info/images/*.jpg"
#
# images=[]
# for filename in glob.glob(info/images):
#     img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#     if img is not None:
#         cv2.imshow("image", img)
#         cv2.waitKey(0)
def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    color_mean = np.mean(image, axis=(0, 1))
    color_std = np.std(image, axis=(0, 1))


    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist / lbp_hist.sum()


    moments = cv2.moments(gray)
    huMoments = cv2.HuMoments(moments).flatten()


    features = np.hstack([color_mean, color_std, lbp_hist, huMoments])
    return features
# le = LabelEncoder()
# y = le.fit_transform(y)

df=pd.read_csv(r'info/train.csv')
df['label'] = df[['healthy', 'scab', 'rust', 'multiple_diseases']].idxmax(axis=1)



X=[]
Y=[]
for _, row in df.iterrows():
    image_file = f"info/images/{row['image_id']}.jpg"
    if os.path.exists(image_file):
        features = extract_features(image_file)
        X.append(features)
        Y.append(row['label'])

X = np.array(X)
le=LabelEncoder()
y = le.fit_transform(Y)

# df=pd.read_csv(r'info/train.csv')
# df['label'] = df[['healthy', 'scab', 'rust', 'multiple_diseases']].idxmax(axis=1)




X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)


voting_clf = VotingClassifier(estimators=[
    ('svm', svm),
    ('rf', rf)
], voting='soft')

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)


print("\n=== Ensemble Model Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Voting Classifier - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


joblib.dump(voting_clf, "ensemble_plant_classifier.pkl")
joblib.dump(scaler, "feature_scaler.pkl")


model = joblib.load("ensemble_plant_classifier.pkl")
scaler = joblib.load("feature_scaler.pkl")


class_names = le.classes_



def predict_image(image_path):
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return

    features = extract_features(image_path)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    predicted_label = class_names[prediction[0]]

    print(f"Prediction for '{os.path.basename(image_path)}': {predicted_label}")



predict_image("info/images/sample_image.jpg")


def predict_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            try:
                predict_image(img_path)
            except Exception as e:
                print(f"Error in {filename}: {e}")


predict_folder("info/test_images")