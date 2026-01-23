/**
 * Utility functions for the Causal Orchestrator frontend
 */

import type { TreatmentEffect, StatusCategory } from '../types';

/**
 * Format a number with specified decimal places
 */
export function formatNumber(value: number, decimals: number = 4): string {
  if (isNaN(value) || !isFinite(value)) {
    return 'N/A';
  }
  return value.toFixed(decimals);
}

/**
 * Format a p-value with appropriate precision
 */
export function formatPValue(pValue: number | null | undefined): string {
  if (pValue === null || pValue === undefined) {
    return 'N/A';
  }
  if (pValue < 0.001) {
    return '< 0.001';
  }
  if (pValue < 0.01) {
    return pValue.toFixed(3);
  }
  return pValue.toFixed(2);
}

/**
 * Format a confidence interval
 */
export function formatCI(lower: number, upper: number, decimals: number = 4): string {
  return `[${formatNumber(lower, decimals)}, ${formatNumber(upper, decimals)}]`;
}

/**
 * Format a percentage
 */
export function formatPercent(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format a duration in milliseconds to human-readable
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  }
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.round((ms % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

/**
 * Format a date to relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: string | Date): string {
  const now = new Date();
  const then = new Date(date);
  const diffMs = now.getTime() - then.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) {
    return 'just now';
  }
  if (diffMins < 60) {
    return `${diffMins} minute${diffMins === 1 ? '' : 's'} ago`;
  }
  if (diffHours < 24) {
    return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
  }
  if (diffDays < 7) {
    return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;
  }
  return then.toLocaleDateString();
}

/**
 * Format a date to ISO date string
 */
export function formatDate(date: string | Date): string {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Get status category from status value
 */
export function getStatusCategory(status: string): StatusCategory {
  if (status === 'pending') return 'pending';
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
  return 'running';
}

/**
 * Get status color for UI
 */
export function getStatusColor(status: string): string {
  const category = getStatusCategory(status);
  switch (category) {
    case 'pending':
      return 'gray';
    case 'running':
      return 'blue';
    case 'completed':
      return 'green';
    case 'failed':
      return 'red';
    default:
      return 'gray';
  }
}

/**
 * Check if a treatment effect is statistically significant
 */
export function isSignificant(effect: TreatmentEffect, alpha: number = 0.05): boolean {
  if (effect.p_value === undefined || effect.p_value === null) {
    // Check if CI excludes zero
    return effect.ci_lower > 0 || effect.ci_upper < 0;
  }
  return effect.p_value < alpha;
}

/**
 * Get the most credible estimate from a list of treatment effects
 */
export function getBestEstimate(effects: TreatmentEffect[]): TreatmentEffect | null {
  if (effects.length === 0) return null;

  // Prefer doubly robust methods
  const dr = effects.find(
    (e) =>
      e.method.toLowerCase().includes('doubly robust') ||
      e.method.toLowerCase().includes('aipw')
  );
  if (dr) return dr;

  // Then prefer matching/IPW
  const matching = effects.find(
    (e) =>
      e.method.toLowerCase().includes('matching') ||
      e.method.toLowerCase().includes('ipw')
  );
  if (matching) return matching;

  // Default to first
  return effects[0];
}

/**
 * Calculate effect size interpretation (Cohen's d-like)
 */
export function interpretEffectSize(estimate: number, stdError: number): string {
  const standardized = Math.abs(estimate) / (stdError * Math.sqrt(2));

  if (standardized < 0.2) return 'negligible';
  if (standardized < 0.5) return 'small';
  if (standardized < 0.8) return 'medium';
  return 'large';
}

/**
 * Validate a Kaggle URL
 */
export function validateKaggleUrl(url: string): string | null {
  if (!url) {
    return 'Kaggle URL is required';
  }
  if (!url.includes('kaggle.com')) {
    return 'Please enter a valid Kaggle dataset URL';
  }
  if (!url.includes('/datasets/')) {
    return 'URL should point to a Kaggle dataset';
  }
  return null;
}

/**
 * Extract dataset name from Kaggle URL
 */
export function extractDatasetName(url: string): string {
  try {
    const match = url.match(/kaggle\.com\/datasets\/([^/]+\/[^/?]+)/);
    if (match) {
      return match[1].replace('/', ' / ');
    }
  } catch {
    // Ignore parsing errors
  }
  return 'Unknown Dataset';
}

/**
 * Generate a unique ID
 */
export function generateId(): string {
  return Math.random().toString(36).substring(2, 10);
}

/**
 * Debounce a function
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Throttle a function
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      fn(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

/**
 * Clamp a number between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Deep clone an object
 */
export function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

/**
 * Check if two arrays are equal (shallow)
 */
export function arraysEqual<T>(a: T[], b: T[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((val, i) => val === b[i]);
}

/**
 * Group an array by a key
 */
export function groupBy<T, K extends string | number>(
  array: T[],
  keyFn: (item: T) => K
): Record<K, T[]> {
  return array.reduce(
    (groups, item) => {
      const key = keyFn(item);
      (groups[key] = groups[key] || []).push(item);
      return groups;
    },
    {} as Record<K, T[]>
  );
}

/**
 * Sort treatment effects by method preference
 */
export function sortEffectsByPreference(effects: TreatmentEffect[]): TreatmentEffect[] {
  const methodOrder = [
    'aipw',
    'doubly robust',
    'ipw',
    'matching',
    'psm',
    'causal forest',
    'double ml',
    't-learner',
    'x-learner',
    's-learner',
    'did',
    'iv',
    'rdd',
    'ols',
  ];

  return [...effects].sort((a, b) => {
    const aMethod = a.method.toLowerCase();
    const bMethod = b.method.toLowerCase();

    const aIndex = methodOrder.findIndex((m) => aMethod.includes(m));
    const bIndex = methodOrder.findIndex((m) => bMethod.includes(m));

    const aOrder = aIndex === -1 ? methodOrder.length : aIndex;
    const bOrder = bIndex === -1 ? methodOrder.length : bIndex;

    return aOrder - bOrder;
  });
}
