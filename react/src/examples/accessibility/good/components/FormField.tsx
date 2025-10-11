import React, { useId } from 'react';

interface FormFieldProps {
  label: string;
  type?: 'text' | 'email' | 'password' | 'tel' | 'url' | 'search';
  placeholder?: string;
  required?: boolean;
  error?: string;
  helpText?: string;
  value?: string;
  onChange?: (value: string) => void;
  onBlur?: () => void;
}

// ✅ 좋은 예시: 접근성이 높은 폼 필드 컴포넌트
const FormField = ({
  label,
  type = 'text',
  placeholder,
  required = false,
  error,
  helpText,
  value,
  onChange,
  onBlur,
}: FormFieldProps) => {
  const fieldId = useId();
  const errorId = useId();
  const helpId = useId();
  
  const describedBy = [
    helpText ? helpId : null,
    error ? errorId : null,
  ].filter(Boolean).join(' ') || undefined;
  
  return (
    <div className="form-field">
      <label 
        htmlFor={fieldId}
        className={`form-label ${required ? 'required' : ''}`}
      >
        {label}
        {required && <span className="required-indicator" aria-label="required">*</span>}
      </label>
      
      <input
        id={fieldId}
        type={type}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        onBlur={onBlur}
        placeholder={placeholder}
        required={required}
        aria-required={required}
        aria-invalid={!!error}
        aria-describedby={describedBy}
        className={`form-input ${error ? 'error' : ''}`}
      />
      
      {helpText && (
        <div id={helpId} className="form-help-text">
          {helpText}
        </div>
      )}
      
      {error && (
        <div 
          id={errorId}
          className="form-error"
          role="alert"
          aria-live="polite"
        >
          <span className="error-icon" aria-hidden="true">⚠️</span>
          {error}
        </div>
      )}
    </div>
  );
};

export default FormField;