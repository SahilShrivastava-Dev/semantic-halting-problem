// ─────────────────────────────────────────────────────────────
// WebSocket event types (mirror backend event schemas)
// ─────────────────────────────────────────────────────────────

export interface BaseEvent {
  type: string
  round?: number
  timestamp?: string
}

export interface RunConfigEvent extends BaseEvent {
  type: 'run_config'
  provider: string
  agent_model: string
  eval_model: string
  scenario_id: string
  topic: string
}

export interface ScenarioStartEvent extends BaseEvent {
  type: 'scenario_start'
  scenario_id: string
  topic: string
  question: string
}

export interface DraftGeneratedEvent extends BaseEvent {
  type: 'draft_generated'
  round: number
  word_count: number
  preview: string
  full_draft: string
}

export interface ISScoreEvent extends BaseEvent {
  type: 'is_score'
  round: number
  score: number
  metrics: {
    faithfulness?: number
    answer_relevancy?: number
    context_precision?: number
    context_recall?: number
  }
  weights?: Record<string, number>
  history?: number[]
  error?: string
}

export interface ConvergenceMetricsEvent extends BaseEvent {
  type: 'convergence_metrics'
  round: number
  distance: number
  threshold: number
  is_gain: number
  dist_history: number[]
}

export interface CriticFeedbackEvent extends BaseEvent {
  type: 'critic_feedback'
  round: number
  feedback: string
  is_approved: boolean
}

export interface HaltSignalEvent extends BaseEvent {
  type: 'halt_signal'
  reason: HaltReason
  round: number
  final_is_score: number
  final_distance: number | null
}

export interface ScenarioCompleteEvent extends BaseEvent {
  type: 'scenario_complete'
  scenario_id: string
  rounds: number
  halt_reason: HaltReason
  final_is_score: number
  is_score_history: number[]
  dist_history: number[]
}

export interface ErrorEvent extends BaseEvent {
  type: 'error'
  message: string
}

export interface EvaluatingEvent extends BaseEvent {
  type: 'evaluating'
  round: number
  message: string
}

export type SHPEvent =
  | RunConfigEvent
  | ScenarioStartEvent
  | DraftGeneratedEvent
  | ISScoreEvent
  | EvaluatingEvent
  | ConvergenceMetricsEvent
  | CriticFeedbackEvent
  | HaltSignalEvent
  | ScenarioCompleteEvent
  | ErrorEvent

export type HaltReason =
  | 'critic_approved'
  | 'entropy_convergence'
  | 'no_information_gain'
  | 'failsafe'
  | ''

// ─────────────────────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────────────────────

export interface RoundDataPoint {
  round: number
  is_score: number
  distance: number | null
  is_gain: number | null
  faithfulness: number | null
  answer_relevancy: number | null
  context_precision: number | null
  context_recall: number | null
}

// One entry per round — powers the ConversationView component.
// Built incrementally as draft_generated → is_score → critic_feedback events arrive.
export interface RoundConversation {
  round: number
  // Writer
  draft: string
  wordCount: number
  // Evaluation (null = still computing)
  isScore: number | null
  isGain: number | null
  evaluating: boolean
  metrics: {
    faithfulness: number | null
    answer_relevancy: number | null
    context_precision: number | null
    context_recall: number | null
  }
  // Critic (null = not yet received)
  criticFeedback: string | null
  isApproved: boolean
}

export interface RunState {
  status: 'idle' | 'connecting' | 'running' | 'complete' | 'error'
  provider: string
  agentModel: string
  evalModel: string
  scenarioId: string
  topic: string
  question: string
  rounds: number
  haltReason: HaltReason
  finalISScore: number
  roundData: RoundDataPoint[]
  conversations: RoundConversation[]
  isWeights: Record<string, number>
  logs: LogEntry[]
  currentDraftPreview: string
  lastCriticFeedback: string
  error: string | null
}

export interface LogEntry {
  id: string
  timestamp: string
  level: 'info' | 'warn' | 'error' | 'success'
  message: string
  round?: number
}

// ─────────────────────────────────────────────────────────────
// API response types
// ─────────────────────────────────────────────────────────────

export interface Scenario {
  id: string
  split: 'train' | 'val' | 'test' | 'custom' | 'demo'
  convergence_profile?: 'fast_critic' | 'entropy_convergence' | 'is_plateau' | 'failsafe'
  topic: string
  question: string
  contexts: string[]
  ground_truth: string
}

export interface ModelsResponse {
  providers: Record<string, string[]>
  defaults: {
    provider: string
    agent_model: Record<string, string>
    eval_model: Record<string, string>
  }
}

export interface WeightsResponse {
  weights: Record<string, number>
  r2_score: number | null
  data_source: string
}

export interface RunConfig {
  provider: string
  agent_model: string
  eval_model: string
  scenario_id: string | null
  topic?: string
  question?: string
  contexts?: string[]
  ground_truth?: string
}

// ─────────────────────────────────────────────────────────────
// Pipeline types
// ─────────────────────────────────────────────────────────────

export interface TopicInfo {
  topic_id: string
  topic: string
  train_count: number
  val_count: number
  test_count: number
}

export interface PipelineQuestionResult {
  phase: 'train' | 'val' | 'test'
  q_idx: number
  question: string
  scenario_id: string
  final_is: number
  halt_reason: string
}

export interface PipelineState {
  status: 'idle' | 'connecting' | 'running' | 'complete' | 'error'
  phase: 'train' | 'val' | 'test' | 'calibrating' | ''
  topicId: string
  trainResults: PipelineQuestionResult[]
  valResults: PipelineQuestionResult[]
  testResults: PipelineQuestionResult[]
  weights: Record<string, number> | null
  r2: number | null
  activeConversation: RoundConversation[]
  activeQuestion: string
  logs: LogEntry[]
  error: string | null
}

// Pipeline WebSocket event interfaces
export interface PipelineStartEvent extends BaseEvent {
  type: 'pipeline_start'
  topic_id: string
  provider: string
  agent_model: string
  eval_model: string
  train_count: number
  val_count: number
  test_count: number
}

export interface PipelineQStartEvent extends BaseEvent {
  type: 'pipeline_q_start'
  phase: 'train' | 'val' | 'test'
  q_idx: number
  total: number
  question: string
  scenario_id: string
}

export interface PipelineQCompleteEvent extends BaseEvent {
  type: 'pipeline_q_complete'
  phase: 'train' | 'val' | 'test'
  q_idx: number
  total: number
  question: string
  scenario_id: string
  final_is: number
  halt_reason: string
}

export interface WeightsLearnedEvent extends BaseEvent {
  type: 'weights_learned'
  weights: Record<string, number>
  r2_score: number | null
  data_source: string
}

export interface CalibratingEvent extends BaseEvent {
  type: 'calibrating'
  message: string
}

export interface PipelineCompleteEvent extends BaseEvent {
  type: 'pipeline_complete'
  topic_id: string
  train_results: PipelineQuestionResult[]
  val_results: PipelineQuestionResult[]
  test_results: PipelineQuestionResult[]
}

export type PipelineEvent =
  | PipelineStartEvent
  | PipelineQStartEvent
  | PipelineQCompleteEvent
  | WeightsLearnedEvent
  | CalibratingEvent
  | PipelineCompleteEvent
  | SHPEvent

export interface PipelineConfig {
  topic_id: string
  provider: string
  agent_model: string
  eval_model: string
}
