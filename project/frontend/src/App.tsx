import React, { useState, useEffect } from 'react';
import { FileText, Presentation, Sparkles, LogOut, Plus, Edit, Download, ThumbsUp, ThumbsDown, MessageSquare, TrendingUp, Save, Trash2, Loader } from 'lucide-react';

const API_URL = 'http://localhost:8000';

interface User {
  email: string;
  user_id: number;
}

interface Project {
  id: number;
  title: string;
  document_type: 'docx' | 'pptx';
  main_topic: string;
  status: string;
  configuration: any;
  created_at: string;
  updated_at: string;
}

interface Content {
  id: number;
  section_id: string;
  title: string;
  content_text: string;
  version: number;
  is_current: boolean;
}

interface Section {
  title: string;
  description: string;
}

// Utility function to format text
const formatText = (text: string) => {
  if (!text) return text;
  let formatted = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/^\* /gm, '• ');
  formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  return formatted;
};

const FormattedText: React.FC<{ text: string; className?: string }> = ({ text, className = '' }) => {
  const formatted = formatText(text);
  return <div className={className} dangerouslySetInnerHTML={{ __html: formatted }} />;
};

export default function OceanAI() {
  const [view, setView] = useState<'login' | 'register' | 'dashboard' | 'project' | 'editor'>('login');
  const [user, setUser] = useState<User | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [contents, setContents] = useState<Content[]>([]);
  
  // Auth states
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Project creation
  const [newProjectTitle, setNewProjectTitle] = useState('');
  const [newProjectType, setNewProjectType] = useState<'docx' | 'pptx'>('docx');
  const [newProjectTopic, setNewProjectTopic] = useState('');
  const [outline, setOutline] = useState<Section[]>([]);
  const [outlineSource, setOutlineSource] = useState<'none' | 'ai' | 'manual'>('none');

  // Refinement
  const [refinementPrompts, setRefinementPrompts] = useState<{[key: string]: string}>({});

  const token = localStorage.getItem('oceanai_token');

  useEffect(() => {
    const savedToken = localStorage.getItem('oceanai_token');
    const savedEmail = localStorage.getItem('oceanai_email');
    const savedUserId = localStorage.getItem('oceanai_user_id');
    
    if (savedToken && savedEmail && savedUserId) {
      setUser({ email: savedEmail, user_id: parseInt(savedUserId) });
      setView('dashboard');
      loadProjects(savedToken);
    }
  }, []);

  const loadProjects = async (authToken: string) => {
    try {
      const response = await fetch(`${API_URL}/api/projects`, {
        headers: { 'Authorization': `Bearer ${authToken}` }
      });
      const data = await response.json();
      setProjects(data.projects || []);
    } catch (error) {
      console.error('Failed to load projects', error);
    }
  };

  const handleAuth = async (isRegister: boolean) => {
    setError('');
    setLoading(true);

    try {
      const endpoint = isRegister ? '/api/auth/register' : '/api/auth/login';
      const body = isRegister 
        ? { email, password, full_name: fullName }
        : { email, password };

      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Authentication failed');
      }

      const data = await response.json();
      
      localStorage.setItem('oceanai_token', data.access_token);
      localStorage.setItem('oceanai_email', data.user_email);
      localStorage.setItem('oceanai_user_id', data.user_id.toString());
      
      setUser({ email: data.user_email, user_id: data.user_id });
      setView('dashboard');
      await loadProjects(data.access_token);
    } catch (err: any) {
      setError(err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('oceanai_token');
    localStorage.removeItem('oceanai_email');
    localStorage.removeItem('oceanai_user_id');
    setUser(null);
    setView('login');
    setProjects([]);
    setCurrentProject(null);
  };

  const resetNewProjectForm = () => {
    setNewProjectTitle('');
    setNewProjectTopic('');
    setNewProjectType('docx');
    setOutline([]);
    setOutlineSource('none');
    setError('');
  };

  const createProject = async () => {
    if (!newProjectTitle || !newProjectTopic || !token) {
      setError('Please fill all fields');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/projects`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          title: newProjectTitle,
          document_type: newProjectType,
          main_topic: newProjectTopic,
        }),
      });

      const data = await response.json();
      const project = data.project;

      if (outline.length > 0) {
        const config = newProjectType === 'docx' 
          ? { sections: outline } 
          : { slides: outline };

        await fetch(`${API_URL}/api/projects/${project.id}/config`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({ configuration: config }),
        });
      }

      await loadProjects(token);
      resetNewProjectForm();
    } catch (err: any) {
      setError(err.message || 'Failed to create project');
    } finally {
      setLoading(false);
    }
  };

  const suggestOutline = async () => {
    if (!newProjectTopic || !token) {
      setError('Please enter a topic first');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      console.log('🚀 Sending request to:', `${API_URL}/api/suggest-outline-direct`);
      console.log('📦 Request body:', { topic: newProjectTopic, document_type: newProjectType });
      
      const response = await fetch(`${API_URL}/api/suggest-outline-direct`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          topic: newProjectTopic,
          document_type: newProjectType
        }),
      });
      
      console.log('📥 Response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('❌ Error response:', errorData);
        throw new Error(errorData.detail || 'Failed to generate outline');
      }

      const data = await response.json();
      console.log('✅ Success response:', data);
      
      // Check if we got suggestions
      if (!data.suggestions || data.suggestions.length === 0) {
        throw new Error('No suggestions received from AI');
      }
      
      console.log(`✨ AI generated ${data.suggestions.length} sections`);
      setOutline(data.suggestions);
      setOutlineSource('ai');
      
    } catch (err: any) {
      console.error('❌ AI suggestion error:', err);
      setError(err.message || 'Failed to generate outline. Please try again.');
      setOutline([]);
      setOutlineSource('none');
    } finally {
      setLoading(false);
    }
  };

  const clearOutline = () => {
    setOutline([]);
    setOutlineSource('none');
  };

  const openProject = async (project: Project) => {
    setCurrentProject(project);
    setLoading(true);
    
    try {
      const response = await fetch(`${API_URL}/api/projects/${project.id}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await response.json();
      setContents(data.contents || []);
      setView('editor');
    } catch (error) {
      console.error('Failed to load project', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteProject = async (projectId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (!window.confirm('Are you sure you want to delete this project? This cannot be undone.')) {
      return;
    }

    if (!token) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/projects/${projectId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) {
        throw new Error('Failed to delete project');
      }

      await loadProjects(token);
    } catch (err: any) {
      setError(err.message || 'Failed to delete project');
    } finally {
      setLoading(false);
    }
  };

  const generateContent = async () => {
    if (!currentProject || !token) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/projects/${currentProject.id}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({}),
      });

      const data = await response.json();
      setContents(data.generated.map((g: any) => ({
        section_id: g.section_id,
        title: g.title,
        content_text: g.content,
        version: 1,
        is_current: true
      })));
    } catch (err: any) {
      setError(err.message || 'Failed to generate content');
    } finally {
      setLoading(false);
    }
  };

  const refineContent = async (sectionId: string) => {
    if (!currentProject || !token) return;
    
    const prompt = refinementPrompts[sectionId];
    if (!prompt) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/projects/${currentProject.id}/refine`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ section_id: sectionId, prompt }),
      });

      const data = await response.json();
      
      setContents(prev => prev.map(c =>
        c.section_id === sectionId
          ? { ...c, content_text: data.content, version: data.version }
          : c
      ));

      setRefinementPrompts(prev => ({ ...prev, [sectionId]: '' }));
    } catch (err: any) {
      setError(err.message || 'Failed to refine content');
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (sectionId: string, feedbackType: 'like' | 'dislike') => {
    if (!currentProject || !token) return;

    try {
      await fetch(`${API_URL}/api/projects/${currentProject.id}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ section_id: sectionId, feedback_type: feedbackType }),
      });
    } catch (error) {
      console.error('Failed to submit feedback', error);
    }
  };

  const exportDocument = async () => {
    if (!currentProject || !token) return;

    try {
      const response = await fetch(`${API_URL}/api/projects/${currentProject.id}/export`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${currentProject.title}.${currentProject.document_type}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      setError('Failed to export document');
    }
  };

  // Auth View
  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <header className="bg-white shadow-sm border-b border-gray-100">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  OceanAI Document Generator
                </h1>
                <p className="text-sm text-gray-600 mt-1">AI-Powered Business Documents</p>
              </div>
            </div>
          </div>
        </header>

        <div className="max-w-md mx-auto px-6 py-12">
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
            <div className="flex gap-2 mb-6">
              <button
                onClick={() => setView('login')}
                className={`flex-1 py-2 rounded-lg font-medium transition-all ${
                  view === 'login'
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                Login
              </button>
              <button
                onClick={() => setView('register')}
                className={`flex-1 py-2 rounded-lg font-medium transition-all ${
                  view === 'register'
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                Register
              </button>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg mb-4">
                {error}
              </div>
            )}

            <form onSubmit={(e) => { e.preventDefault(); handleAuth(view === 'register'); }} className="space-y-4">
              {view === 'register' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="John Doe"
                  />
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="email@example.com"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="••••••••"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-blue-200 transition-all disabled:opacity-50"
              >
                {loading ? 'Processing...' : (view === 'login' ? 'Login' : 'Register')}
              </button>
            </form>
          </div>
        </div>
      </div>
    );
  }

  // Dashboard View
  if (view === 'dashboard') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <header className="bg-white shadow-sm border-b border-gray-100">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl">
                  <Sparkles className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    My Projects
                  </h1>
                  <p className="text-sm text-gray-600 mt-1">{user.email}</p>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
              >
                <LogOut className="w-5 h-5" />
                Logout
              </button>
            </div>
          </div>
        </header>

        <div className="max-w-7xl mx-auto px-6 py-8">
          {/* Create Project */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8 mb-8">
            <div className="flex items-center gap-3 mb-6">
              <Plus className="w-8 h-8 text-blue-600" />
              <h2 className="text-2xl font-bold text-gray-900">Create New Project</h2>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg mb-4">
                {error}
              </div>
            )}

            <form className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Project Title</label>
                <input
                  type="text"
                  value={newProjectTitle}
                  onChange={(e) => setNewProjectTitle(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Q4 Business Plan 2025"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <button
                  type="button"
                  onClick={() => setNewProjectType('docx')}
                  className={`flex items-center justify-center gap-3 p-6 rounded-xl border-2 transition-all ${
                    newProjectType === 'docx'
                      ? 'border-blue-600 bg-blue-50'
                      : 'border-gray-300 hover:border-blue-400'
                  }`}
                >
                  <FileText className="w-8 h-8 text-blue-600" />
                  <span className="font-semibold">Word Document</span>
                </button>
                <button
                  type="button"
                  onClick={() => setNewProjectType('pptx')}
                  className={`flex items-center justify-center gap-3 p-6 rounded-xl border-2 transition-all ${
                    newProjectType === 'pptx'
                      ? 'border-purple-600 bg-purple-50'
                      : 'border-gray-300 hover:border-purple-400'
                  }`}
                >
                  <Presentation className="w-8 h-8 text-purple-600" />
                  <span className="font-semibold">PowerPoint</span>
                </button>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Main Topic / Description</label>
                <textarea
                  value={newProjectTopic}
                  onChange={(e) => setNewProjectTopic(e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent h-24"
                  placeholder="Describe what this document is about..."
                />
              </div>

              <button
                type="button"
                onClick={suggestOutline}
                disabled={loading || !newProjectTopic}
                className="w-full py-3 bg-purple-50 text-purple-700 font-medium rounded-lg hover:bg-purple-100 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
              >
                <Sparkles className="w-5 h-5" />
                {loading ? 'Generating...' : 'AI Suggest Outline'}
              </button>

              {outline.length > 0 && (
                <div className={`border rounded-xl p-4 ${
                  outlineSource === 'ai' 
                    ? 'bg-gradient-to-br from-purple-50 to-blue-50 border-purple-200' 
                    : 'bg-gray-50 border-gray-200'
                }`}>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-gray-900 flex items-center gap-2">
                      {outlineSource === 'ai' ? '✨ AI-Generated Outline' : '📝 Custom Outline'}
                    </h3>
                    {outlineSource === 'ai' && (
                      <span className="px-3 py-1 bg-purple-600 text-white text-xs font-medium rounded-full flex items-center gap-1">
                        <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
                        AI Powered
                      </span>
                    )}
                  </div>
                  
                  <div className="space-y-2">
                    {outline.map((item, idx) => (
                      <div 
                        key={idx} 
                        className={`p-3 rounded-lg border ${
                          outlineSource === 'ai'
                            ? 'bg-white border-purple-200 hover:border-purple-300'
                            : 'bg-white border-gray-200 hover:border-gray-300'
                        } transition-colors`}
                      >
                        <p className="font-medium text-gray-900">{item.title}</p>
                        <p className="text-sm text-gray-600 mt-1">{item.description}</p>
                      </div>
                    ))}
                  </div>
                  
                  <button
                    type="button"
                    onClick={clearOutline}
                    className="mt-3 text-sm text-red-600 hover:text-red-800 font-medium flex items-center gap-1"
                  >
                    <span>×</span> Clear Outline
                  </button>
                </div>
              )}

              <button
                type="button"
                onClick={createProject}
                disabled={loading}
                className="w-full py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-blue-200 transition-all disabled:opacity-50"
              >
                {loading ? 'Creating...' : 'Create Project'}
              </button>
            </form>
          </div>

          {/* Projects List */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project) => (
              <div
                key={project.id}
                className="bg-white rounded-xl shadow-lg border border-gray-100 p-6 hover:shadow-xl transition-shadow relative group"
              >
                <button
                  onClick={(e) => deleteProject(project.id, e)}
                  className="absolute top-4 right-4 p-2 bg-red-50 text-red-600 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-100"
                  title="Delete project"
                >
                  <Trash2 className="w-5 h-5" />
                </button>

                <div onClick={() => openProject(project)} className="cursor-pointer">
                  <div className="flex items-start justify-between mb-4">
                    <div className="p-3 bg-gradient-to-br from-blue-100 to-purple-100 rounded-lg">
                      {project.document_type === 'docx' ? (
                        <FileText className="w-8 h-8 text-blue-600" />
                      ) : (
                        <Presentation className="w-8 h-8 text-purple-600" />
                      )}
                    </div>
                  </div>
                  <h3 className="text-lg font-bold text-gray-900 mb-2">{project.title}</h3>
                  <p className="text-sm text-gray-600 mb-4 line-clamp-2">{project.main_topic}</p>
                  <div className="text-xs text-gray-500">
                    {new Date(project.created_at).toLocaleDateString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Editor View
  if (view === 'editor' && currentProject) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <header className="bg-white shadow-sm border-b border-gray-100">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <button
                onClick={() => setView('dashboard')}
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                ← Back to Dashboard
              </button>
              <h1 className="text-2xl font-bold text-gray-900">{currentProject.title}</h1>
              <button
                onClick={exportDocument}
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg hover:shadow-lg transition-all"
              >
                <Download className="w-5 h-5" />
                Export
              </button>
            </div>
          </div>
        </header>

        <div className="max-w-7xl mx-auto px-6 py-8">
          {contents.length === 0 ? (
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-12 text-center">
              <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-6">No content generated yet</p>
              <button
                onClick={generateContent}
                disabled={loading}
                className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-blue-200 transition-all disabled:opacity-50 inline-flex items-center gap-2"
              >
                {loading ? <Loader className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5" />}
                {loading ? 'Generating...' : 'Generate Content'}
              </button>
            </div>
          ) : (
            <div className="space-y-6">
              {contents.map((content) => (
                <div key={content.section_id} className="bg-white rounded-2xl shadow-lg border border-gray-100 p-8">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-xl font-bold text-gray-900">{content.title}</h3>
                      <p className="text-sm text-gray-500">Version {content.version}</p>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => submitFeedback(content.section_id, 'like')}
                        className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                      >
                        <ThumbsUp className="w-5 h-5" />
                      </button>
                      <button
                        onClick={() => submitFeedback(content.section_id, 'dislike')}
                        className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      >
                        <ThumbsDown className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-6 mb-4">
                    <FormattedText text={content.content_text} className="text-gray-800 whitespace-pre-wrap" />
                  </div>

                  <div className="flex gap-3">
                    <input
                      type="text"
                      value={refinementPrompts[content.section_id] || ''}
                      onChange={(e) => setRefinementPrompts(prev => ({
                        ...prev,
                        [content.section_id]: e.target.value
                      }))}
                      className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="E.g., 'Make more formal' or 'Add bullet points'"
                    />
                    <button
                      onClick={() => refineContent(content.section_id)}
                      disabled={loading || !refinementPrompts[content.section_id]}
                      className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium rounded-lg hover:shadow-lg transition-all disabled:opacity-50 flex items-center gap-2"
                    >
                      {loading ? <Loader className="w-5 h-5 animate-spin" /> : <Edit className="w-5 h-5" />}
                      Refine
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  return null;
}