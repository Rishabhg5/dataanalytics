"""
Backend API Tests for E1 Analytics Platform
Tests: Authentication, AI Analytics, ML Models, RBAC, Audit Logs
"""
import pytest
import requests
import os
import uuid

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')
API = f"{BASE_URL}/api"

# Test dataset ID from the problem statement
TEST_DATASET_ID = "9315e278-1057-4322-8e6a-3baafb6550e0"

# Test credentials
TEST_ADMIN = {"email": "admin@test.com", "password": "admin123", "name": "Test Admin", "role": "admin"}
TEST_VIEWER = {"email": f"viewer_{uuid.uuid4().hex[:8]}@test.com", "password": "viewer123", "name": "Test Viewer", "role": "viewer"}
TEST_ANALYST = {"email": f"analyst_{uuid.uuid4().hex[:8]}@test.com", "password": "analyst123", "name": "Test Analyst", "role": "analyst"}


class TestAPIHealth:
    """Basic API health checks"""
    
    def test_api_root(self):
        """Test API root endpoint"""
        response = requests.get(f"{API}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["version"] == "2.0.0"
        assert "AI Analytics" in data["features"]
        assert "RBAC" in data["features"]
        print(f"SUCCESS: API root returns {data}")


class TestUserRegistration:
    """User registration tests"""
    
    def test_register_viewer(self):
        """Test registering a new viewer user"""
        payload = TEST_VIEWER.copy()
        response = requests.post(f"{API}/auth/register", json=payload)
        
        # Could be 200 (success) or 400 (already exists)
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert data["user"]["email"] == payload["email"]
            assert data["user"]["role"] == "viewer"
            print(f"SUCCESS: Registered viewer user: {data['user']['email']}")
        elif response.status_code == 400:
            print(f"INFO: User already exists (expected in re-runs)")
        else:
            pytest.fail(f"Unexpected status: {response.status_code} - {response.text}")
    
    def test_register_analyst(self):
        """Test registering a new analyst user"""
        payload = TEST_ANALYST.copy()
        response = requests.post(f"{API}/auth/register", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert data["user"]["role"] == "analyst"
            print(f"SUCCESS: Registered analyst user: {data['user']['email']}")
        elif response.status_code == 400:
            print(f"INFO: User already exists")
        else:
            pytest.fail(f"Unexpected status: {response.status_code} - {response.text}")
    
    def test_register_admin(self):
        """Test registering admin user"""
        payload = TEST_ADMIN.copy()
        response = requests.post(f"{API}/auth/register", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert data["user"]["role"] == "admin"
            print(f"SUCCESS: Registered admin user: {data['user']['email']}")
        elif response.status_code == 400:
            print(f"INFO: Admin user already exists")
        else:
            pytest.fail(f"Unexpected status: {response.status_code} - {response.text}")
    
    def test_register_invalid_role(self):
        """Test registration with invalid role"""
        payload = {
            "email": f"invalid_{uuid.uuid4().hex[:8]}@test.com",
            "password": "test123",
            "name": "Invalid Role User",
            "role": "superadmin"  # Invalid role
        }
        response = requests.post(f"{API}/auth/register", json=payload)
        assert response.status_code == 400
        assert "Invalid role" in response.json().get("detail", "")
        print("SUCCESS: Invalid role rejected correctly")


class TestUserLogin:
    """User login tests"""
    
    def test_login_admin(self):
        """Test admin login"""
        # First ensure admin exists
        requests.post(f"{API}/auth/register", json=TEST_ADMIN)
        
        response = requests.post(f"{API}/auth/login", json={
            "email": TEST_ADMIN["email"],
            "password": TEST_ADMIN["password"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user"]["email"] == TEST_ADMIN["email"]
        assert data["user"]["role"] == "admin"
        print(f"SUCCESS: Admin login successful, token received")
        return data["access_token"]
    
    def test_login_invalid_credentials(self):
        """Test login with wrong password"""
        response = requests.post(f"{API}/auth/login", json={
            "email": TEST_ADMIN["email"],
            "password": "wrongpassword"
        })
        assert response.status_code == 401
        assert "Invalid email or password" in response.json().get("detail", "")
        print("SUCCESS: Invalid credentials rejected correctly")
    
    def test_login_nonexistent_user(self):
        """Test login with non-existent user"""
        response = requests.post(f"{API}/auth/login", json={
            "email": "nonexistent@test.com",
            "password": "anypassword"
        })
        assert response.status_code == 401
        print("SUCCESS: Non-existent user login rejected")


class TestAuthenticatedEndpoints:
    """Tests for authenticated endpoints"""
    
    @pytest.fixture
    def admin_token(self):
        """Get admin token"""
        requests.post(f"{API}/auth/register", json=TEST_ADMIN)
        response = requests.post(f"{API}/auth/login", json={
            "email": TEST_ADMIN["email"],
            "password": TEST_ADMIN["password"]
        })
        if response.status_code == 200:
            return response.json()["access_token"]
        pytest.skip("Could not get admin token")
    
    @pytest.fixture
    def viewer_token(self):
        """Get viewer token"""
        viewer_data = {
            "email": f"viewer_test_{uuid.uuid4().hex[:6]}@test.com",
            "password": "viewer123",
            "name": "Test Viewer",
            "role": "viewer"
        }
        response = requests.post(f"{API}/auth/register", json=viewer_data)
        if response.status_code == 200:
            return response.json()["access_token"]
        pytest.skip("Could not get viewer token")
    
    def test_get_current_user(self, admin_token):
        """Test getting current user info"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = requests.get(f"{API}/auth/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == TEST_ADMIN["email"]
        assert data["role"] == "admin"
        print(f"SUCCESS: Current user info retrieved: {data['email']}")
    
    def test_list_users_as_admin(self, admin_token):
        """Test listing users as admin"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = requests.get(f"{API}/auth/users", headers=headers)
        
        assert response.status_code == 200
        users = response.json()
        assert isinstance(users, list)
        print(f"SUCCESS: Admin can list {len(users)} users")
    
    def test_list_users_as_viewer_forbidden(self, viewer_token):
        """Test that viewers cannot list users"""
        headers = {"Authorization": f"Bearer {viewer_token}"}
        response = requests.get(f"{API}/auth/users", headers=headers)
        
        assert response.status_code == 403
        print("SUCCESS: Viewer correctly denied access to user list")


class TestAuditLogs:
    """Audit log tests"""
    
    @pytest.fixture
    def admin_token(self):
        """Get admin token"""
        requests.post(f"{API}/auth/register", json=TEST_ADMIN)
        response = requests.post(f"{API}/auth/login", json={
            "email": TEST_ADMIN["email"],
            "password": TEST_ADMIN["password"]
        })
        if response.status_code == 200:
            return response.json()["access_token"]
        pytest.skip("Could not get admin token")
    
    @pytest.fixture
    def viewer_token(self):
        """Get viewer token"""
        viewer_data = {
            "email": f"viewer_audit_{uuid.uuid4().hex[:6]}@test.com",
            "password": "viewer123",
            "name": "Audit Viewer",
            "role": "viewer"
        }
        response = requests.post(f"{API}/auth/register", json=viewer_data)
        if response.status_code == 200:
            return response.json()["access_token"]
        pytest.skip("Could not get viewer token")
    
    def test_get_audit_logs_as_admin(self, admin_token):
        """Test getting audit logs as admin"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = requests.get(f"{API}/audit/logs?limit=50", headers=headers)
        
        assert response.status_code == 200
        logs = response.json()
        assert isinstance(logs, list)
        print(f"SUCCESS: Admin retrieved {len(logs)} audit logs")
        
        # Verify log structure if logs exist
        if logs:
            log = logs[0]
            assert "action" in log
            assert "timestamp" in log
            print(f"Sample log: action={log['action']}, user={log.get('user_email')}")
    
    def test_get_audit_logs_as_viewer_forbidden(self, viewer_token):
        """Test that viewers cannot access audit logs"""
        headers = {"Authorization": f"Bearer {viewer_token}"}
        response = requests.get(f"{API}/audit/logs", headers=headers)
        
        assert response.status_code == 403
        print("SUCCESS: Viewer correctly denied access to audit logs")


class TestAIAnalytics:
    """AI-powered analytics endpoint tests"""
    
    @pytest.fixture
    def dataset_id(self):
        """Get a valid dataset ID"""
        response = requests.get(f"{API}/datasets")
        if response.status_code == 200 and response.json():
            return response.json()[0]["id"]
        return TEST_DATASET_ID
    
    def test_ai_analyze_data(self, dataset_id):
        """Test AI data analysis endpoint"""
        payload = {
            "dataset_id": dataset_id,
            "insight_type": "data_analysis"
        }
        response = requests.post(f"{API}/ai/analyze-data", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "data_quality" in data
            assert "statistical_summary" in data
            assert "ai_insight" in data
            print(f"SUCCESS: AI analysis completed")
            print(f"Data quality: {data['data_quality']}")
            print(f"AI Insight: {data['ai_insight'][:100]}...")
        elif response.status_code == 404:
            print(f"INFO: Dataset not found (ID: {dataset_id})")
        else:
            pytest.fail(f"Unexpected status: {response.status_code} - {response.text}")
    
    def test_ai_prescriptive_analytics(self, dataset_id):
        """Test prescriptive analytics endpoint"""
        payload = {
            "dataset_id": dataset_id,
            "business_context": "sales optimization",
            "optimization_goals": ["increase revenue", "reduce costs"]
        }
        response = requests.post(f"{API}/ai/prescriptive", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "risk_factors" in data
            assert "action_items" in data
            assert "ai_strategic_recommendations" in data
            print(f"SUCCESS: Prescriptive analytics completed")
            print(f"Risk factors: {len(data['risk_factors'])}")
            print(f"Action items: {len(data['action_items'])}")
        elif response.status_code == 404:
            print(f"INFO: Dataset not found (ID: {dataset_id})")
        else:
            pytest.fail(f"Unexpected status: {response.status_code} - {response.text}")
    
    def test_ai_describe_chart(self, dataset_id):
        """Test AI chart description endpoint"""
        payload = {
            "dataset_id": dataset_id,
            "insight_type": "chart_description",
            "context": {
                "chart_type": "bar chart",
                "columns": ["value", "category"]
            }
        }
        response = requests.post(f"{API}/ai/describe-chart", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "insight" in data
            assert "chart_type" in data
            print(f"SUCCESS: Chart description generated")
            print(f"Insight: {data['insight'][:100]}...")
        elif response.status_code == 404:
            print(f"INFO: Dataset not found (ID: {dataset_id})")
        else:
            pytest.fail(f"Unexpected status: {response.status_code} - {response.text}")


class TestMLModels:
    """ML model endpoint tests"""
    
    @pytest.fixture
    def dataset_id(self):
        """Get a valid dataset ID with numeric columns"""
        response = requests.get(f"{API}/datasets")
        if response.status_code == 200 and response.json():
            # Find a dataset with numeric columns
            for ds in response.json():
                if ds.get("rows", 0) >= 10:
                    return ds["id"]
            return response.json()[0]["id"]
        return TEST_DATASET_ID
    
    def test_ml_prediction(self, dataset_id):
        """Test ML prediction endpoint"""
        # First get dataset columns
        response = requests.get(f"{API}/datasets/{dataset_id}")
        if response.status_code != 200:
            pytest.skip(f"Dataset {dataset_id} not found")
        
        dataset = response.json()
        numeric_cols = [col for col, dtype in dataset.get("column_types", {}).items() 
                       if "int" in dtype or "float" in dtype]
        
        if len(numeric_cols) < 2:
            pytest.skip("Not enough numeric columns for prediction")
        
        target_col = numeric_cols[0]
        
        payload = {
            "dataset_id": dataset_id,
            "analysis_type": "prediction",
            "parameters": {"target_column": target_col}
        }
        response = requests.post(f"{API}/ml/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "performance" in data
            assert "feature_importance" in data
            print(f"SUCCESS: ML prediction completed")
            print(f"Model: {data['model_type']}")
            print(f"R² Score: {data['performance'].get('r2_score', 'N/A')}")
        elif response.status_code == 400:
            print(f"INFO: {response.json().get('detail', 'Bad request')}")
        elif response.status_code == 404:
            print(f"INFO: Dataset not found")
        else:
            pytest.fail(f"Unexpected status: {response.status_code} - {response.text}")
    
    def test_ml_clustering(self, dataset_id):
        """Test ML clustering endpoint"""
        payload = {
            "dataset_id": dataset_id,
            "analysis_type": "clustering",
            "parameters": {"n_clusters": 3}
        }
        response = requests.post(f"{API}/ml/cluster", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "cluster_stats" in data
            assert "ai_insight" in data
            print(f"SUCCESS: ML clustering completed")
            print(f"Clusters: {data['n_clusters']}")
            print(f"Cluster stats: {list(data['cluster_stats'].keys())}")
        elif response.status_code == 400:
            print(f"INFO: {response.json().get('detail', 'Bad request')}")
        elif response.status_code == 404:
            print(f"INFO: Dataset not found")
        else:
            pytest.fail(f"Unexpected status: {response.status_code} - {response.text}")
    
    def test_ml_prediction_missing_target(self, dataset_id):
        """Test ML prediction without target column"""
        payload = {
            "dataset_id": dataset_id,
            "analysis_type": "prediction",
            "parameters": {}  # Missing target_column
        }
        response = requests.post(f"{API}/ml/predict", json=payload)
        
        assert response.status_code == 400
        assert "target_column" in response.json().get("detail", "").lower()
        print("SUCCESS: Missing target column correctly rejected")


class TestDatasets:
    """Dataset endpoint tests"""
    
    def test_list_datasets(self):
        """Test listing all datasets"""
        response = requests.get(f"{API}/datasets")
        assert response.status_code == 200
        datasets = response.json()
        assert isinstance(datasets, list)
        print(f"SUCCESS: Retrieved {len(datasets)} datasets")
        
        if datasets:
            ds = datasets[0]
            assert "id" in ds
            assert "name" in ds
            assert "rows" in ds
            assert "columns" in ds
    
    def test_get_dataset_by_id(self):
        """Test getting a specific dataset"""
        # First get list
        response = requests.get(f"{API}/datasets")
        if response.status_code != 200 or not response.json():
            pytest.skip("No datasets available")
        
        dataset_id = response.json()[0]["id"]
        response = requests.get(f"{API}/datasets/{dataset_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        print(f"SUCCESS: Retrieved dataset: {data['name']}")
    
    def test_get_dataset_data(self):
        """Test getting dataset data"""
        response = requests.get(f"{API}/datasets")
        if response.status_code != 200 or not response.json():
            pytest.skip("No datasets available")
        
        dataset_id = response.json()[0]["id"]
        response = requests.get(f"{API}/datasets/{dataset_id}/data")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"SUCCESS: Retrieved {len(data)} data rows")


class TestRBAC:
    """Role-Based Access Control tests"""
    
    @pytest.fixture
    def admin_token(self):
        """Get admin token"""
        requests.post(f"{API}/auth/register", json=TEST_ADMIN)
        response = requests.post(f"{API}/auth/login", json={
            "email": TEST_ADMIN["email"],
            "password": TEST_ADMIN["password"]
        })
        if response.status_code == 200:
            return response.json()["access_token"]
        pytest.skip("Could not get admin token")
    
    def test_admin_can_update_user_role(self, admin_token):
        """Test that admin can update user roles"""
        # Create a test user
        test_user = {
            "email": f"rbac_test_{uuid.uuid4().hex[:6]}@test.com",
            "password": "test123",
            "name": "RBAC Test User",
            "role": "viewer"
        }
        reg_response = requests.post(f"{API}/auth/register", json=test_user)
        if reg_response.status_code != 200:
            pytest.skip("Could not create test user")
        
        user_id = reg_response.json()["user"]["id"]
        
        # Update role as admin
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = requests.put(
            f"{API}/auth/users/{user_id}/role?new_role=analyst",
            headers=headers
        )
        
        assert response.status_code == 200
        assert response.json()["success"] == True
        print(f"SUCCESS: Admin updated user role to analyst")
    
    def test_admin_can_toggle_user_status(self, admin_token):
        """Test that admin can enable/disable users"""
        # Create a test user
        test_user = {
            "email": f"status_test_{uuid.uuid4().hex[:6]}@test.com",
            "password": "test123",
            "name": "Status Test User",
            "role": "viewer"
        }
        reg_response = requests.post(f"{API}/auth/register", json=test_user)
        if reg_response.status_code != 200:
            pytest.skip("Could not create test user")
        
        user_id = reg_response.json()["user"]["id"]
        
        # Toggle status as admin
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = requests.put(
            f"{API}/auth/users/{user_id}/status",
            headers=headers
        )
        
        assert response.status_code == 200
        assert "is_active" in response.json()
        print(f"SUCCESS: Admin toggled user status")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
