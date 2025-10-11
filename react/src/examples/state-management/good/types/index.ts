export interface Todo {
  id: number;
  text: string;
  completed: boolean;
  category: string;
  priority: 'low' | 'medium' | 'high';
  createdAt: Date;
}

export interface TodoFilter {
  status: 'all' | 'active' | 'completed';
  searchTerm: string;
  category: string;
}

export interface TodoStats {
  total: number;
  completed: number;
  active: number;
}

export interface TodoState {
  todos: Todo[];
  filter: TodoFilter;
  isLoading: boolean;
  error: string | null;
}