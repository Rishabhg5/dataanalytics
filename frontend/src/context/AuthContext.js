import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  // 1. Initialize user from localStorage immediately to prevent "flash" of null on reload
  const [user, setUser] = useState(() => {
    try {
      const storedUser = localStorage.getItem('user');
      return storedUser ? JSON.parse(storedUser) : null;
    } catch (error) {
      return null;
    }
  });
  
  const [token, setToken] = useState(localStorage.getItem('token'));
  
  // Start loading as true only if we have a token but no user yet
  // If we have a user from localStorage, we can start with loading = false (or keep true to verify in bg)
  // For smoothest UI, we start true, but since user is already set, Layout won't redirect immediately.
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initializeAuth = async () => {
      if (token) {
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        try {
          // Verify token is still valid by fetching fresh user data
          const response = await axios.get(`${API}/auth/me`);
          setUser(response.data);
          // Update local storage with fresh data
          localStorage.setItem('user', JSON.stringify(response.data));
        } catch (error) {
          console.error('Failed to fetch user:', error);
          // If token is invalid (401), clear everything
          logout();
        }
      } else {
        // No token, ensure user is cleared
        setUser(null);
        localStorage.removeItem('user');
      }
      setLoading(false);
    };

    initializeAuth();
  }, [token]);

  const login = async (email, password) => {
    const response = await axios.post(`${API}/auth/login`, { email, password });
    const { access_token, user: userData } = response.data;
    
    // Save Token AND User to localStorage
    localStorage.setItem('token', access_token);
    localStorage.setItem('user', JSON.stringify(userData));
    
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
    setToken(access_token);
    setUser(userData);
    
    return userData;
  };

  const register = async (email, password, name, role = 'viewer') => {
    const response = await axios.post(`${API}/auth/register`, { 
      email, 
      password, 
      name, 
      role 
    });
    const { access_token, user: userData } = response.data;
    
    // Save Token AND User to localStorage
    localStorage.setItem('token', access_token);
    localStorage.setItem('user', JSON.stringify(userData));

    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
    setToken(access_token);
    setUser(userData);
    
    return userData;
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user'); // Clear user data
    delete axios.defaults.headers.common['Authorization'];
    setToken(null);
    setUser(null);
  };

  const hasPermission = (permission) => {
    if (!user) return false;
    
    const permissions = {
      admin: ['read', 'write', 'delete', 'manage_users', 'view_audit', 'mask_data', 'export'],
      manager: ['read', 'write', 'delete', 'view_audit', 'export'],
      analyst: ['read', 'write', 'export'],
      viewer: ['read']
    };
    
    return permissions[user.role]?.includes(permission) || false;
  };

  const isAdmin = () => user?.role === 'admin';
  const isManager = () => user?.role === 'manager' || user?.role === 'admin';

  return (
    <AuthContext.Provider value={{ 
      user, 
      token, 
      loading, 
      login, 
      register, 
      logout, 
      hasPermission,
      isAdmin,
      isManager,
      isAuthenticated: !!user 
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default AuthContext;