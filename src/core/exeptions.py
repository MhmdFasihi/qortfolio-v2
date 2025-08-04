# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

"""
Custom Exception Classes for Qortfolio V2
Location: src/core/exceptions.py

Comprehensive exception hierarchy for better error handling throughout the system.
"""

from typing import Optional, Dict, Any


class QortfolioError(Exception):
    """Base exception class for all Qortfolio V2 errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize base Qortfolio error.
        
        Args:
            message: Error message
            error_code: Optional error code for programmatic handling
            context: Optional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        return base_msg


# ==================== CONFIGURATION EXCEPTIONS ====================

class ConfigurationError(QortfolioError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_file: Optional[str] = None, 
                 missing_key: Optional[str] = None):
        context = {}
        if config_file:
            context['config_file'] = config_file
        if missing_key:
            context['missing_key'] = missing_key
        
        super().__init__(message, error_code="CONFIG_ERROR", context=context)
        self.config_file = config_file
        self.missing_key = missing_key


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, missing_key: str, config_file: Optional[str] = None):
        message = f"Missing required configuration: {missing_key}"
        if config_file:
            message += f" in {config_file}"
        
        super().__init__(message, config_file=config_file, missing_key=missing_key)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""
    
    def __init__(self, key: str, value: Any, expected_type: str):
        message = f"Invalid configuration for '{key}': got {type(value).__name__}, expected {expected_type}"
        super().__init__(message, missing_key=key)
        self.value = value
        self.expected_type = expected_type


# ==================== DATA EXCEPTIONS ====================

class DataError(QortfolioError):
    """Base class for data-related errors."""
    pass


class DataCollectionError(DataError):
    """Raised when data collection fails."""
    
    def __init__(self, message: str, source: Optional[str] = None, 
                 symbol: Optional[str] = None, response_code: Optional[int] = None):
        context = {}
        if source:
            context['source'] = source
        if symbol:
            context['symbol'] = symbol
        if response_code:
            context['response_code'] = response_code
            
        super().__init__(message, error_code="DATA_COLLECTION_ERROR", context=context)
        self.source = source
        self.symbol = symbol
        self.response_code = response_code


class DataValidationError(DataError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Any = None, validation_rule: Optional[str] = None):
        context = {
            'field': field,
            'value': str(value) if value is not None else None,
            'validation_rule': validation_rule
        }
        
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", context=context)
        self.field = field
        self.value = value
        self.validation_rule = validation_rule


class EmptyDataError(DataError):
    """Raised when expected data is empty or None."""
    
    def __init__(self, data_type: str, source: Optional[str] = None):
        message = f"No data received for {data_type}"
        if source:
            message += f" from {source}"
            
        super().__init__(message, error_code="EMPTY_DATA_ERROR")
        self.data_type = data_type
        self.source = source


class DataIntegrityError(DataError):
    """Raised when data integrity checks fail."""
    
    def __init__(self, message: str, expected_count: Optional[int] = None, 
                 actual_count: Optional[int] = None):
        context = {}
        if expected_count is not None:
            context['expected_count'] = expected_count
        if actual_count is not None:
            context['actual_count'] = actual_count
            
        super().__init__(message, error_code="DATA_INTEGRITY_ERROR", context=context)
        self.expected_count = expected_count
        self.actual_count = actual_count


# ==================== API EXCEPTIONS ====================

class APIError(QortfolioError):
    """Base class for API-related errors."""
    pass


class APIConnectionError(APIError):
    """Raised when API connection fails."""
    
    def __init__(self, api_name: str, endpoint: Optional[str] = None, 
                 status_code: Optional[int] = None):
        message = f"Failed to connect to {api_name} API"
        if endpoint:
            message += f" at {endpoint}"
        if status_code:
            message += f" (HTTP {status_code})"
            
        context = {
            'api_name': api_name,
            'endpoint': endpoint,
            'status_code': status_code
        }
        
        super().__init__(message, error_code="API_CONNECTION_ERROR", context=context)
        self.api_name = api_name
        self.endpoint = endpoint
        self.status_code = status_code


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, api_name: str, retry_after: Optional[int] = None):
        message = f"Rate limit exceeded for {api_name} API"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
            
        super().__init__(message, error_code="API_RATE_LIMIT_ERROR")
        self.api_name = api_name
        self.retry_after = retry_after


class APIResponseError(APIError):
    """Raised when API returns invalid response."""
    
    def __init__(self, api_name: str, message: str, response_data: Any = None):
        full_message = f"{api_name} API error: {message}"
        
        context = {'api_name': api_name}
        if response_data:
            context['response_data'] = str(response_data)
            
        super().__init__(full_message, error_code="API_RESPONSE_ERROR", context=context)
        self.api_name = api_name
        self.response_data = response_data


# ==================== CALCULATION EXCEPTIONS ====================

class CalculationError(QortfolioError):
    """Base class for calculation-related errors."""
    pass


class TimeCalculationError(CalculationError):
    """Raised when time calculations fail."""
    
    def __init__(self, message: str, current_time: Any = None, expiry_time: Any = None):
        context = {
            'current_time': str(current_time) if current_time else None,
            'expiry_time': str(expiry_time) if expiry_time else None
        }
        
        super().__init__(message, error_code="TIME_CALCULATION_ERROR", context=context)
        self.current_time = current_time
        self.expiry_time = expiry_time


class MathematicalError(CalculationError):
    """Raised when mathematical calculations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 parameters: Optional[Dict] = None):
        context = {'operation': operation}
        if parameters:
            context.update(parameters)
            
        super().__init__(message, error_code="MATHEMATICAL_ERROR", context=context)
        self.operation = operation
        self.parameters = parameters


class InvalidParameterError(CalculationError):
    """Raised when calculation parameters are invalid."""
    
    def __init__(self, parameter_name: str, value: Any, constraint: str):
        message = f"Invalid parameter '{parameter_name}': {value} ({constraint})"
        
        context = {
            'parameter_name': parameter_name,
            'value': str(value),
            'constraint': constraint
        }
        
        super().__init__(message, error_code="INVALID_PARAMETER_ERROR", context=context)
        self.parameter_name = parameter_name
        self.value = value
        self.constraint = constraint


# ==================== OPTIONS EXCEPTIONS ====================

class OptionsError(QortfolioError):
    """Base class for options-related errors."""
    pass


class OptionsDataError(OptionsError):
    """Raised when options data is invalid or missing."""
    
    def __init__(self, message: str, instrument: Optional[str] = None, 
                 strike: Optional[float] = None, expiry: Optional[str] = None):
        context = {
            'instrument': instrument,
            'strike': strike,
            'expiry': expiry
        }
        
        super().__init__(message, error_code="OPTIONS_DATA_ERROR", context=context)
        self.instrument = instrument
        self.strike = strike
        self.expiry = expiry


class OptionsPricingError(OptionsError):
    """Raised when options pricing calculations fail."""
    
    def __init__(self, message: str, model: Optional[str] = None, 
                 parameters: Optional[Dict] = None):
        context = {'model': model}
        if parameters:
            context.update(parameters)
            
        super().__init__(message, error_code="OPTIONS_PRICING_ERROR", context=context)
        self.model = model
        self.parameters = parameters


class GreeksCalculationError(OptionsError):
    """Raised when Greeks calculations fail."""
    
    def __init__(self, message: str, greek_type: Optional[str] = None, 
                 spot: Optional[float] = None, strike: Optional[float] = None):
        context = {
            'greek_type': greek_type,
            'spot': spot,
            'strike': strike
        }
        
        super().__init__(message, error_code="GREEKS_CALCULATION_ERROR", context=context)
        self.greek_type = greek_type
        self.spot = spot
        self.strike = strike


# ==================== DASHBOARD EXCEPTIONS ====================

class DashboardError(QortfolioError):
    """Base class for dashboard-related errors."""
    pass


class DashboardRenderError(DashboardError):
    """Raised when dashboard rendering fails."""
    
    def __init__(self, message: str, component: Optional[str] = None, 
                 page: Optional[str] = None):
        context = {'component': component, 'page': page}
        
        super().__init__(message, error_code="DASHBOARD_RENDER_ERROR", context=context)
        self.component = component
        self.page = page


class DataDisplayError(DashboardError):
    """Raised when data display fails in dashboard."""
    
    def __init__(self, message: str, data_type: Optional[str] = None, 
                 widget: Optional[str] = None):
        context = {'data_type': data_type, 'widget': widget}
        
        super().__init__(message, error_code="DATA_DISPLAY_ERROR", context=context)
        self.data_type = data_type
        self.widget = widget


# ==================== UTILITY FUNCTIONS ====================

def handle_exception(func):
    """Decorator to handle exceptions and convert them to QortfolioError."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except QortfolioError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Convert other exceptions to QortfolioError
            raise QortfolioError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={'function': func.__name__, 'original_error': str(e)}
            ) from e
    return wrapper


def get_error_details(error: Exception) -> Dict[str, Any]:
    """Extract detailed information from an exception."""
    details = {
        'error_type': type(error).__name__,
        'message': str(error)
    }
    
    if isinstance(error, QortfolioError):
        details.update({
            'error_code': error.error_code,
            'context': error.context
        })
    
    return details


# ==================== EXPORTS ====================

__all__ = [
    # Base exceptions
    'QortfolioError',
    
    # Configuration exceptions
    'ConfigurationError',
    'MissingConfigurationError', 
    'InvalidConfigurationError',
    
    # Data exceptions
    'DataError',
    'DataCollectionError',
    'DataValidationError',
    'EmptyDataError',
    'DataIntegrityError',
    
    # API exceptions
    'APIError',
    'APIConnectionError',
    'APIRateLimitError',
    'APIResponseError',
    
    # Calculation exceptions
    'CalculationError',
    'TimeCalculationError',
    'MathematicalError',
    'InvalidParameterError',
    
    # Options exceptions
    'OptionsError',
    'OptionsDataError',
    'OptionsPricingError',
    'GreeksCalculationError',
    
    # Dashboard exceptions
    'DashboardError',
    'DashboardRenderError',
    'DataDisplayError',
    
    # Utility functions
    'handle_exception',
    'get_error_details'
]