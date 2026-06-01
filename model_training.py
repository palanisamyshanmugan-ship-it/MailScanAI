import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

training_data = []

spam_emails = [
"Win a free iPhone today",
"Congratulations you won a lottery",
"Cheap medicines available now",
"Limited time prize claim",
"Get rich quickly",
"Earn money from home",
"Click to receive your reward",
"Exclusive offer just for you",
"Huge discount on products",
"Buy now and save",
"Claim your cash prize",
"You are selected as winner",
"Instant loan approved",
"Free vacation package",
"Hot deals available today",
"Weight loss pills cheap",
"Special coupon expires soon",
"Act now limited stock",
"Promotional offer available",
"Free gift card waiting",
"Best online casino bonus",
"Luxury watch discount",
"Mega sale starts today",
"Lowest prices guaranteed",
"Special marketing campaign"
]

phishing_emails = [
"Your bank account has been suspended",
"Verify your PayPal password immediately",
"Netflix account requires verification",
"Update your banking details",
"Your account has unusual activity",
"Click here to avoid suspension",
"Security alert login required",
"Confirm your identity now",
"Reset your password urgently",
"Unauthorized login detected",
"Verify credit card information",
"Your payment failed update details",
"Bank verification required",
"PayPal security notification",
"Account locked login now",
"Validate your credentials",
"Tax refund claim login",
"Update billing information",
"Microsoft account alert",
"Amazon account verification",
"Confirm account ownership",
"Suspicious login attempt",
"Recover your account now",
"Identity verification needed",
"Immediate action required"
]

business_emails = [
"Meeting scheduled for tomorrow",
"Please review attached proposal",
"Quarterly financial report",
"Client presentation update",
"Project deadline reminder",
"Invoice attached for review",
"Team meeting agenda",
"Contract approval request",
"Business development update",
"Partnership discussion",
"Budget review meeting",
"Vendor payment confirmation",
"Sales performance report",
"Employee onboarding update",
"Project status report",
"Customer feedback summary",
"Marketing strategy meeting",
"Executive review session",
"Internal audit report",
"Board meeting notes",
"Procurement update",
"Operational review document",
"Monthly KPI report",
"Stakeholder discussion",
"Business proposal attached"
]

personal_emails = [
"Happy birthday my friend",
"Let's meet for dinner",
"Family vacation photos",
"How are you doing",
"Weekend trip plans",
"Congratulations on your achievement",
"Thank you for your help",
"See you soon",
"Wedding invitation",
"Let's catch up",
"Photos from our trip",
"Hope you are well",
"Dinner tonight",
"Family gathering details",
"Missing you a lot",
"Call me when free",
"Happy anniversary wishes",
"Picnic plans this weekend",
"Good luck for exam",
"Get well soon",
"Looking forward to meeting",
"Coffee tomorrow morning",
"Friends reunion event",
"Personal invitation",
"Holiday greetings"
]

for item in spam_emails:
    training_data.append((item, "Spam"))

for item in phishing_emails:
    training_data.append((item, "Phishing"))

for item in business_emails:
    training_data.append((item, "Business"))

for item in personal_emails:
    training_data.append((item, "Personal"))

texts = [x[0] for x in training_data]
labels = [x[1] for x in training_data]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=8000
)

X = vectorizer.fit_transform(texts)

model = LogisticRegression(
    C=5.0,
    max_iter=1000
)

model.fit(X, encoded_labels)

os.makedirs("model", exist_ok=True)

with open("model/email_classifier.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "vectorizer": vectorizer
    }, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model training completed successfully.")