# report_agent/pipeline/stages/validator.py
"""
ResultValidator stage: Validates LLM analysis against actual data.

This stage is responsible for:
- Validating that significance claims are justified by the data
- Checking that required fields are present in structured output
- Preventing over-interpretation and hallucinations
"""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

from report_agent.pipeline.models import RawAnalysis, ValidatedResult

log = logging.getLogger(__name__)


class ResultValidator:
    """
    Validates LLM analysis results against actual data.
    
    This prevents over-interpretation by checking that:
    - Significance claims (HIGH/MEDIUM) are backed by statistical evidence
    - Changes are not within normal variation
    - Required structured output fields are present
    
    Usage:
        validator = ResultValidator()
        result = validator.validate(raw_analysis, original_df)
    """
    
    def validate(self, analysis: RawAnalysis, df: pd.DataFrame) -> ValidatedResult:
        """
        Validate LLM analysis against actual data.
        
        Args:
            analysis: RawAnalysis from the LLMAnalyzer stage
            df: Original DataFrame used for analysis
            
        Returns:
            ValidatedResult with validation status and any warnings
        """
        warnings: List[str] = []
        errors: List[str] = []
        
        structured = analysis.structured
        
        # Check for structured output
        if not structured:
            errors.append("No structured output found")
            return ValidatedResult(
                model_name=analysis.model_name,
                narrative=analysis.narrative,
                structured={},
                validation_status="errors",
                validation_warnings=errors,
                artifacts=analysis.artifacts,
                df=df,
            )
        
        significance = structured.get("significance", "").upper()
        
        # Validate significance claims against data
        if "date" in df.columns and "value" in df.columns:
            data_warnings = self._validate_significance_claims(df, significance, structured)
            warnings.extend(data_warnings)
        
        # Check significance is valid
        if not significance or significance not in ["HIGH", "MEDIUM", "LOW", "NONE"]:
            errors.append("Missing or invalid significance assessment (must be HIGH/MEDIUM/LOW/NONE)")
        
        status = "errors" if errors else ("warnings" if warnings else "valid")
        
        return ValidatedResult(
            model_name=analysis.model_name,
            narrative=analysis.narrative,
            structured=structured,
            validation_status=status,
            validation_warnings=warnings + errors,
            artifacts=analysis.artifacts,
            df=df,
        )
    
    def _validate_significance_claims(
        self,
        df: pd.DataFrame,
        significance: str,
        structured: dict
    ) -> List[str]:
        """
        Validate that significance assessment is justified by the data.
        
        Returns a list of warning messages.
        """
        warnings = []
        
        try:
            # Aggregate by week
            df_copy = df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
            df_copy = df_copy.dropna(subset=["date"])
            
            if len(df_copy) == 0:
                return warnings
            
            df_copy["week"] = df_copy["date"].dt.to_period("W")
            
            # Handle multiple rows per date (aggregate by date first, then by week)
            if "label" in df_copy.columns:
                # Group by date and label, sum values, then group by week
                daily = df_copy.groupby(["date", "label"])["value"].sum().reset_index()
                daily["week"] = daily["date"].dt.to_period("W")
                weekly = daily.groupby("week")["value"].sum().reset_index()
            else:
                daily = df_copy.groupby("date")["value"].sum().reset_index()
                daily["week"] = daily["date"].dt.to_period("W")
                weekly = daily.groupby("week")["value"].sum().reset_index()
            
            if len(weekly) < 4:
                return warnings  # Not enough data to validate
            
            last_week = weekly.iloc[-1]["value"]
            prev_week = weekly.iloc[-2]["value"] if len(weekly) >= 2 else None
            four_week_values = weekly.iloc[-4:]["value"]
            four_week_avg = four_week_values.mean()
            four_week_std = four_week_values.std()
            
            if prev_week is None or four_week_std == 0:
                return warnings
            
            wow_change_pct = ((last_week - prev_week) / prev_week) * 100 if prev_week != 0 else 0
            std_devs_away = (last_week - four_week_avg) / four_week_std
            
            # Validate significance assessment
            if significance == "HIGH":
                # HIGH should have strong evidence
                if abs(wow_change_pct) < 15:
                    warnings.append(
                        f"HIGH significance but only {wow_change_pct:.1f}% change (expected >15%)"
                    )
                if abs(std_devs_away) < 2:
                    warnings.append(
                        f"HIGH significance but only {std_devs_away:.1f} std devs from avg (expected >2)"
                    )
            
            if significance == "MEDIUM":
                # MEDIUM should have some evidence
                if abs(wow_change_pct) < 5:
                    warnings.append(
                        f"MEDIUM significance but only {wow_change_pct:.1f}% change (expected >5%)"
                    )
            
            # Check if within normal variation
            if abs(std_devs_away) < 1 and significance in ["HIGH", "MEDIUM"]:
                warnings.append(
                    f"Significance {significance} but change is within normal variation (±1 std dev)"
                )
            
            # Check if it's trend continuation (not unusual)
            if len(weekly) >= 3:
                trend_direction = "increasing" if weekly.iloc[-1]["value"] > weekly.iloc[-2]["value"] else "decreasing"
                prev_trend = "increasing" if weekly.iloc[-2]["value"] > weekly.iloc[-3]["value"] else "decreasing"
                if trend_direction == prev_trend and significance == "HIGH":
                    warnings.append(
                        f"HIGH significance but change continues existing {trend_direction} trend (may not be unusual)"
                    )
                    
        except Exception as e:
            # Don't fail validation on calculation errors, just log
            log.debug(f"Error calculating validation statistics: {e}")
        
        return warnings
