from utils.extractor import FeatureExtractor

extractor = FeatureExtractor()

left = [1.0] * 63
right = [2.0] * 63

features = extractor.extract(left, right)

print("Total Features:", len(features))
print(features[:10])