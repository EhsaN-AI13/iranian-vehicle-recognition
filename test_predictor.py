from predictor import VehiclePredictor


predictor = VehiclePredictor()

results = predictor.predict(
    "test_images/car4.jpg"
)

print("\nPrediction Results:")
print("=" * 40)

for result in results:

    print(
        f"{result['class']:<20}"
        f"{result['confidence'] * 100:.2f}%"
    )