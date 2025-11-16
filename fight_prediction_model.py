import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class UFCFightPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        
    def load_real_fighters(self):
        makhachev = {
            'record': 27, 'losses': 1, 'wr': 27/28,
            'grappling': 0.95, 'striking': 0.595,
            'sub_rate': 13/27, 'ko_rate': 5/27,
            'td_acc': 0.537, 'td_def': 0.909,
            'weight_exp': 0.5, 'age': 34,
            'reach': 70.5, 'height': 70,
            'recent_form': 0.95, 'opp_quality': 0.92,
            'champ_exp': 1.0, 'control_time': 0.45,
            'strike_def': 0.62, 'strike_acc': 0.595,
            'strikes_per_min': 2.6, 'td_per_15min': 3.2,
            'win_streak': 15, 'ufc_fights': 18, 'ufc_record': 16
        }
        della_maddalena = {
            'record': 18, 'losses': 2, 'wr': 18/20,
            'grappling': 0.50, 'striking': 0.90,
            'sub_rate': 2/18, 'ko_rate': 12/18,
            'td_acc': 0.35, 'td_def': 0.70,
            'weight_exp': 1.0, 'age': 29,
            'reach': 73, 'height': 72,
            'recent_form': 0.90, 'opp_quality': 0.85,
            'champ_exp': 0.8, 'control_time': 0.15,
            'strike_def': 0.58, 'strike_acc': 0.55,
            'strikes_per_min': 6.8, 'td_per_15min': 0.16,
            'win_streak': 18, 'ufc_fights': 8, 'ufc_record': 8
        }
        return makhachev, della_maddalena
    
    def load_from_csv(self, filename='ufc_fights.csv'):
        return pd.read_csv(filename)
    
    def train(self, df):
        X = df.drop('outcome', axis=1)
        y = df['outcome']
        X_scaled = self.scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=250, max_depth=15, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=250, max_depth=7, random_state=42)
        rf.fit(X_scaled, y)
        gb.fit(X_scaled, y)
        self.model = {'rf': rf, 'gb': gb}
        return self.model
    
    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        rf_pred = self.model['rf'].predict(X_test_scaled)
        gb_pred = self.model['gb'].predict(X_test_scaled)
        ensemble_pred = (rf_pred + gb_pred) / 2
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        acc = accuracy_score(y_test, ensemble_pred)
        return acc, ensemble_pred
    
    def prepare_features(self, f1, f2):
        return np.array([
            f1['record'], f1['losses'], f1['wr'], f1['grappling'], f1['striking'],
            f1['sub_rate'], f1['ko_rate'], f1['td_acc'], f1['td_def'], f1['weight_exp'],
            f1['age'], f1['reach'], f1['height'], f1['recent_form'], f1['opp_quality'],
            f1['champ_exp'], f1['control_time'], f1['strike_def'], f1['strike_acc'],
            f1['win_streak'], f1['ufc_fights'], f1['ufc_record'],
            f2['record'], f2['losses'], f2['wr'], f2['grappling'], f2['striking'],
            f2['sub_rate'], f2['ko_rate'], f2['td_acc'], f2['td_def'], f2['weight_exp'],
            f2['age'], f2['reach'], f2['height'], f2['recent_form'], f2['opp_quality'],
            f2['champ_exp'], f2['control_time'], f2['strike_def'], f2['strike_acc'],
            f2['win_streak'], f2['ufc_fights'], f2['ufc_record'],
            f1['record'] - f2['record'], f1['wr'] - f2['wr'], f1['grappling'] - f2['grappling'],
            f1['striking'] - f2['striking'], f1['td_acc'] - f2['td_acc'], f1['td_def'] - f2['td_def'],
            f1['weight_exp'] - f2['weight_exp'], f1['age'] - f2['age'],
            f1['reach'] - f2['reach'], f1['height'] - f2['height'],
            f1['recent_form'] - f2['recent_form'], f1['opp_quality'] - f2['opp_quality'],
            f1['champ_exp'] - f2['champ_exp'], f1['control_time'] - f2['control_time'],
            f1['strike_def'] - f2['strike_def'], f1['strike_acc'] - f2['strike_acc'],
            f1['win_streak'] - f2['win_streak'], f1['ufc_fights'] - f2['ufc_fights'],
            f1['ufc_record'] - f2['ufc_record']
        ]).reshape(1, -1)
    
    def predict(self, f1, f2):
        features = self.prepare_features(f1, f2)
        features_scaled = self.scaler.transform(features)
        rf_pred = self.model['rf'].predict_proba(features_scaled)[0]
        gb_pred = self.model['gb'].predict_proba(features_scaled)[0]
        f1_prob = (rf_pred[1] + gb_pred[1]) / 2
        f2_prob = (rf_pred[0] + gb_pred[0]) / 2
        return f1_prob, f2_prob
    
    def analyze(self, f1, f2):
        f1_prob, f2_prob = self.predict(f1, f2)
        winner = 'Makhachev' if f1_prob > f2_prob else 'Della Maddalena'
        print(f"Winner: {winner} ({max(f1_prob, f2_prob):.1%})")

def main():
    predictor = UFCFightPredictor()
    
    print("Loading fight data from CSV...")
    df = predictor.load_from_csv('ufc_fights.csv')
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training...")
    predictor.train(pd.concat([X_train, y_train], axis=1))
    
    print("Testing...")
    acc, _ = predictor.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.3f}\n")
    
    f1, f2 = predictor.load_real_fighters()
    predictor.analyze(f1, f2)

if __name__ == "__main__":
    main()
