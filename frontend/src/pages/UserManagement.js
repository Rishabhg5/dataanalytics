import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';
import { Users, Shield, ShieldCheck, ShieldX, UserCog, AlertTriangle } from 'lucide-react';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ROLE_COLORS = {
  admin: 'bg-purple-100 text-purple-800 border-purple-200',
  manager: 'bg-blue-100 text-blue-800 border-blue-200',
  analyst: 'bg-green-100 text-green-800 border-green-200',
  viewer: 'bg-slate-100 text-slate-800 border-slate-200'
};

const ROLE_ICONS = {
  admin: ShieldCheck,
  manager: Shield,
  analyst: UserCog,
  viewer: Users
};

export default function UserManagement() {
  const { user, isAdmin, isManager } = useAuth();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedUser, setSelectedUser] = useState(null);
  const [newRole, setNewRole] = useState('');

  useEffect(() => {
    if (isAdmin() || isManager()) {
      fetchUsers();
    }
  }, []);

  const fetchUsers = async () => {
    try {
      const response = await axios.get(`${API}/auth/users`);
      setUsers(response.data);
    } catch (error) {
      console.error('Error fetching users:', error);
      toast.error('Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  const handleRoleChange = async (userId, role) => {
    try {
      await axios.put(`${API}/auth/users/${userId}/role?new_role=${role}`);
      toast.success('Role updated successfully');
      fetchUsers();
      setSelectedUser(null);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to update role');
    }
  };

  const handleToggleStatus = async (userId) => {
    try {
      await axios.put(`${API}/auth/users/${userId}/status`);
      toast.success('User status updated');
      fetchUsers();
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to update status');
    }
  };

  if (!isAdmin() && !isManager()) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-8 text-center">
          <AlertTriangle className="w-12 h-12 text-amber-600 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-amber-900 mb-2">Access Restricted</h2>
          <p className="text-amber-700">You need admin or manager privileges to access this page.</p>
        </div>
      </div>
    );
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight mb-4">
          User Management
        </h1>
        <p className="text-lg text-slate-600 leading-relaxed">
          Manage user accounts, roles, and permissions. {isAdmin() ? 'Full admin access.' : 'Manager view - limited to lower roles.'}
        </p>
      </div>

      {/* Role Legend */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
        <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-4">Role Hierarchy</h3>
        <div className="flex flex-wrap gap-4">
          {Object.entries(ROLE_COLORS).map(([role, classes]) => {
            const Icon = ROLE_ICONS[role];
            return (
              <div key={role} className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${classes}`}>
                <Icon className="w-4 h-4" />
                <span className="text-sm font-medium capitalize">{role}</span>
              </div>
            );
          })}
        </div>
        <p className="mt-3 text-sm text-slate-500">
          Admin → Manager → Analyst → Viewer (highest to lowest permissions)
        </p>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <p className="text-slate-600">Loading users...</p>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50">
                <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">User</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Role</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Status</th>
                <th className="px-6 py-4 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Created</th>
                {isAdmin() && (
                  <th className="px-6 py-4 text-right text-xs font-medium text-slate-700 uppercase tracking-wider">Actions</th>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {users.map((u) => {
                const RoleIcon = ROLE_ICONS[u.role] || Users;
                const isCurrentUser = u.id === user?.id;
                
                return (
                  <tr key={u.id} className={`hover:bg-slate-50 ${isCurrentUser ? 'bg-indigo-50/50' : ''}`}>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center">
                          <span className="text-indigo-700 font-semibold">
                            {u.name?.charAt(0).toUpperCase() || 'U'}
                          </span>
                        </div>
                        <div>
                          <p className="font-medium text-slate-900">
                            {u.name}
                            {isCurrentUser && <span className="ml-2 text-xs text-indigo-600">(You)</span>}
                          </p>
                          <p className="text-sm text-slate-500">{u.email}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      {selectedUser === u.id && isAdmin() && !isCurrentUser ? (
                        <select
                          value={newRole || u.role}
                          onChange={(e) => setNewRole(e.target.value)}
                          className="h-9 px-3 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        >
                          <option value="viewer">Viewer</option>
                          <option value="analyst">Analyst</option>
                          <option value="manager">Manager</option>
                          <option value="admin">Admin</option>
                        </select>
                      ) : (
                        <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${ROLE_COLORS[u.role]}`}>
                          <RoleIcon className="w-3.5 h-3.5" />
                          {u.role}
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium ${
                        u.is_active !== false
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${u.is_active !== false ? 'bg-green-600' : 'bg-red-600'}`} />
                        {u.is_active !== false ? 'Active' : 'Disabled'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-slate-600 text-sm">
                      {formatDate(u.created_at)}
                    </td>
                    {isAdmin() && (
                      <td className="px-6 py-4">
                        <div className="flex items-center justify-end gap-2">
                          {selectedUser === u.id ? (
                            <>
                              <button
                                onClick={() => handleRoleChange(u.id, newRole || u.role)}
                                className="px-3 py-1.5 bg-indigo-600 text-white rounded-lg text-sm hover:bg-indigo-700"
                              >
                                Save
                              </button>
                              <button
                                onClick={() => { setSelectedUser(null); setNewRole(''); }}
                                className="px-3 py-1.5 bg-slate-100 text-slate-700 rounded-lg text-sm hover:bg-slate-200"
                              >
                                Cancel
                              </button>
                            </>
                          ) : (
                            <>
                              {!isCurrentUser && (
                                <>
                                  <button
                                    onClick={() => { setSelectedUser(u.id); setNewRole(u.role); }}
                                    className="px-3 py-1.5 bg-slate-100 text-slate-700 rounded-lg text-sm hover:bg-slate-200"
                                  >
                                    Edit Role
                                  </button>
                                  <button
                                    onClick={() => handleToggleStatus(u.id)}
                                    className={`px-3 py-1.5 rounded-lg text-sm ${
                                      u.is_active !== false
                                        ? 'bg-red-50 text-red-600 hover:bg-red-100'
                                        : 'bg-green-50 text-green-600 hover:bg-green-100'
                                    }`}
                                  >
                                    {u.is_active !== false ? 'Disable' : 'Enable'}
                                  </button>
                                </>
                              )}
                            </>
                          )}
                        </div>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
