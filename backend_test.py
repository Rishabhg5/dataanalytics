import requests
import sys
import json
import io
import pandas as pd
from datetime import datetime

class AnalyticsAPITester:
    def __init__(self, base_url="https://analytix-forge.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.uploaded_dataset_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {}
        
        if files is None:
            headers['Content-Type'] = 'application/json'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files)
                else:
                    response = requests.post(url, json=data, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    return success, response.json()
                except:
                    return success, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    print(f"   Response: {response.text}")
                except:
                    pass
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def create_sample_csv(self):
        """Create a sample CSV file for testing"""
        data = {
            'id': range(1, 101),
            'sales': [100 + i * 2.5 + (i % 10) * 5 for i in range(100)],
            'profit': [20 + i * 0.5 + (i % 7) * 2 for i in range(100)],
            'region': ['North', 'South', 'East', 'West'] * 25,
            'date': [f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(100)]
        }
        df = pd.DataFrame(data)
        
        # Add some missing values for testing cleaning operations
        df.loc[5:10, 'sales'] = None
        df.loc[15:18, 'profit'] = None
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode('utf-8')

    def test_root_endpoint(self):
        """Test API root endpoint"""
        success, response = self.run_test(
            "API Root",
            "GET",
            "",
            200
        )
        return success

    def test_upload_dataset(self):
        """Test dataset upload"""
        csv_data = self.create_sample_csv()
        files = {'file': ('test_data.csv', csv_data, 'text/csv')}
        
        success, response = self.run_test(
            "Upload Dataset",
            "POST",
            "datasets/upload",
            200,
            files=files
        )
        
        if success and 'id' in response:
            self.uploaded_dataset_id = response['id']
            print(f"   Dataset ID: {self.uploaded_dataset_id}")
            print(f"   Rows: {response.get('rows', 'N/A')}")
            print(f"   Columns: {response.get('columns', 'N/A')}")
        
        return success

    def test_get_datasets(self):
        """Test getting all datasets"""
        success, response = self.run_test(
            "Get All Datasets",
            "GET",
            "datasets",
            200
        )
        
        if success:
            print(f"   Found {len(response)} datasets")
        
        return success

    def test_get_dataset_detail(self):
        """Test getting specific dataset with data"""
        if not self.uploaded_dataset_id:
            print("❌ No dataset ID available for testing")
            return False
            
        success, response = self.run_test(
            "Get Dataset Detail",
            "GET",
            f"datasets/{self.uploaded_dataset_id}?limit=50",
            200
        )
        
        if success:
            print(f"   Dataset rows: {response.get('total_rows', 'N/A')}")
            print(f"   Preview rows: {len(response.get('data', []))}")
        
        return success

    def test_data_cleaning(self):
        """Test data cleaning operations"""
        if not self.uploaded_dataset_id:
            print("❌ No dataset ID available for cleaning tests")
            return False
        
        # Test remove duplicates
        success1, _ = self.run_test(
            "Remove Duplicates",
            "POST",
            f"datasets/{self.uploaded_dataset_id}/clean",
            200,
            data={
                "dataset_id": self.uploaded_dataset_id,
                "operation": "remove_duplicates"
            }
        )
        
        # Test fill missing values
        success2, _ = self.run_test(
            "Fill Missing Values (Mean)",
            "POST",
            f"datasets/{self.uploaded_dataset_id}/clean",
            200,
            data={
                "dataset_id": self.uploaded_dataset_id,
                "operation": "fill_missing",
                "column": "sales",
                "parameters": {"method": "mean"}
            }
        )
        
        return success1 and success2

    def test_descriptive_analytics(self):
        """Test descriptive analytics"""
        if not self.uploaded_dataset_id:
            print("❌ No dataset ID available for analytics tests")
            return False
            
        success, response = self.run_test(
            "Descriptive Analytics",
            "POST",
            "analytics/descriptive",
            200,
            data={
                "dataset_id": self.uploaded_dataset_id,
                "analysis_type": "descriptive"
            }
        )
        
        if success and 'statistics' in response:
            stats = response['statistics']
            print(f"   Analyzed {len(stats)} numeric columns")
            for col, stat in list(stats.items())[:2]:  # Show first 2 columns
                print(f"   {col}: mean={stat.get('mean', 0):.2f}, std={stat.get('std', 0):.2f}")
        
        return success

    def test_trend_analysis(self):
        """Test trend analysis"""
        if not self.uploaded_dataset_id:
            print("❌ No dataset ID available for trend analysis")
            return False
            
        success, response = self.run_test(
            "Trend Analysis",
            "POST",
            "analytics/trends",
            200,
            data={
                "dataset_id": self.uploaded_dataset_id,
                "analysis_type": "trends",
                "parameters": {"column": "sales"}
            }
        )
        
        if success:
            print(f"   Trend: {response.get('trend', 'N/A')}")
            print(f"   Slope: {response.get('slope', 0):.4f}")
            print(f"   R²: {response.get('r_squared', 0):.3f}")
        
        return success

    def test_anomaly_detection(self):
        """Test anomaly detection"""
        if not self.uploaded_dataset_id:
            print("❌ No dataset ID available for anomaly detection")
            return False
            
        success, response = self.run_test(
            "Anomaly Detection",
            "POST",
            "analytics/anomalies",
            200,
            data={
                "dataset_id": self.uploaded_dataset_id,
                "analysis_type": "anomalies",
                "columns": ["sales", "profit"],
                "parameters": {"contamination": 0.1}
            }
        )
        
        if success:
            print(f"   Total anomalies: {response.get('total_anomalies', 0)}")
            print(f"   Anomaly percentage: {response.get('anomaly_percentage', 0):.2f}%")
        
        return success

    def test_forecasting(self):
        """Test forecasting"""
        if not self.uploaded_dataset_id:
            print("❌ No dataset ID available for forecasting")
            return False
            
        success, response = self.run_test(
            "Forecasting",
            "POST",
            "analytics/forecast",
            200,
            data={
                "dataset_id": self.uploaded_dataset_id,
                "analysis_type": "forecast",
                "parameters": {"column": "sales", "periods": 5}
            }
        )
        
        if success:
            print(f"   Historical points: {len(response.get('historical', []))}")
            print(f"   Forecast points: {len(response.get('forecast', []))}")
            if response.get('forecast'):
                print(f"   First forecast value: {response['forecast'][0]:.2f}")
        
        return success

    def test_delete_dataset(self):
        """Test dataset deletion"""
        if not self.uploaded_dataset_id:
            print("❌ No dataset ID available for deletion test")
            return False
            
        success, response = self.run_test(
            "Delete Dataset",
            "DELETE",
            f"datasets/{self.uploaded_dataset_id}",
            200
        )
        
        return success

def main():
    print("🚀 Starting E1 Analytics API Testing")
    print("=" * 50)
    
    tester = AnalyticsAPITester()
    
    # Run all tests in sequence
    tests = [
        tester.test_root_endpoint,
        tester.test_upload_dataset,
        tester.test_get_datasets,
        tester.test_get_dataset_detail,
        tester.test_data_cleaning,
        tester.test_descriptive_analytics,
        tester.test_trend_analysis,
        tester.test_anomaly_detection,
        tester.test_forecasting,
        tester.test_delete_dataset,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
    
    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())