use std::io::{self, BufRead, Write};

use santorini_core::{
    board::FullGameState,
    gods::PartialAction,
    search::{SearchContext, get_past_win_search_terminator, negamax_search},
    search_terminators::DynamicNodesVisitedSearchTerminator,
    transposition_table::TranspositionTable,
    utils::find_action_path,
};
use serde::Deserialize;
use serde_json::{Value, json};

const DEFAULT_NODES: usize = 20_000;
const DEFAULT_TOP_K: usize = 8;

#[derive(Debug, Deserialize)]
struct Request {
    command: String,
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    fen: Option<String>,
    #[serde(default)]
    nodes: Option<usize>,
    #[serde(default)]
    top_k: Option<usize>,
}

fn parse_state(request: &Request) -> Result<FullGameState, String> {
    let fen = request
        .fen
        .as_ref()
        .ok_or_else(|| "request is missing fen".to_owned())?;
    FullGameState::try_from(fen).map_err(|error| format!("invalid FEN: {error}"))
}

fn analyze(request: &Request, tt: &mut TranspositionTable) -> Result<Value, String> {
    let state = parse_state(request)?;
    if state.board.get_winner().is_some() {
        return Err("cannot analyze a terminal position".to_owned());
    }

    let node_limit = request.nodes.unwrap_or(DEFAULT_NODES);
    if node_limit == 0 {
        return Err("nodes must be greater than zero".to_owned());
    }

    let terminator = DynamicNodesVisitedSearchTerminator::new(node_limit);
    let mut context = SearchContext::new(tt, terminator);
    let result = negamax_search(
        &mut context,
        state.clone(),
        get_past_win_search_terminator(),
    );
    let best = result
        .best_move
        .ok_or_else(|| "search completed without a best move".to_owned())?;
    let actions = find_action_path(&state, &best.child_state).unwrap_or_default();

    Ok(json!({
        "command": "analyze",
        "fen": request.fen,
        "requested_nodes": node_limit,
        "nodes_visited": result.nodes_visited,
        "completed_depth": result.last_fully_completed_depth,
        "best_move": {
            "action": best.action_str,
            "actions": actions,
            "next_fen": best.child_state,
            "score": best.score,
            "depth": best.depth,
            "nodes_visited": best.nodes_visited,
            "trigger": best.trigger,
        },
    }))
}

fn legal_moves(request: &Request) -> Result<Value, String> {
    let state = parse_state(request)?;
    if state.board.get_winner().is_some() {
        return Ok(json!({
            "command": "legal_moves",
            "fen": request.fen,
            "moves": [],
            "terminal": true,
        }));
    }

    let moves: Vec<Value> = state
        .get_next_states_interactive()
        .into_iter()
        .map(|next| {
            let no_moves = next
                .actions
                .iter()
                .any(|action| matches!(action, PartialAction::NoMoves));
            json!({
                "actions": next.actions,
                "next_fen": next.state,
                "no_moves": no_moves,
            })
        })
        .collect();

    Ok(json!({
        "command": "legal_moves",
        "fen": request.fen,
        "moves": moves,
        "terminal": false,
    }))
}

fn analyze_root_moves(request: &Request, tt: &mut TranspositionTable) -> Result<Value, String> {
    let state = parse_state(request)?;
    if state.board.get_winner().is_some() {
        return Err("cannot analyze a terminal position".to_owned());
    }
    let nodes_per_move = request.nodes.unwrap_or(DEFAULT_NODES);
    if nodes_per_move == 0 {
        return Err("nodes must be greater than zero".to_owned());
    }
    let top_k = request.top_k.unwrap_or(DEFAULT_TOP_K);
    if top_k == 0 {
        return Err("top_k must be greater than zero".to_owned());
    }

    let mut ranked = Vec::new();
    let mut total_nodes_visited = 0usize;
    for next in state.get_next_states_interactive() {
        let no_moves = next
            .actions
            .iter()
            .any(|action| matches!(action, PartialAction::NoMoves));
        if no_moves {
            continue;
        }

        let (score, completed_depth, nodes_visited, child_best_score) =
            if next.state.board.get_winner().is_some() {
                (santorini_core::search::WINNING_SCORE, 0usize, 0usize, None)
            } else {
                let terminator = DynamicNodesVisitedSearchTerminator::new(nodes_per_move);
                let mut context = SearchContext::new(tt, terminator);
                let result = negamax_search(
                    &mut context,
                    next.state.clone(),
                    get_past_win_search_terminator(),
                );
                let child_score = result
                    .best_move
                    .as_ref()
                    .map(|best| best.score)
                    .ok_or_else(|| "child search completed without a best move".to_owned())?;
                total_nodes_visited += result.nodes_visited;
                (
                    -child_score,
                    result.last_fully_completed_depth + 1,
                    result.nodes_visited,
                    Some(child_score),
                )
            };

        ranked.push((
            score,
            next.state.to_string(),
            json!({
                "actions": next.actions,
                "next_fen": next.state,
                "score": score,
                "child_score": child_best_score,
                "completed_depth": completed_depth,
                "nodes_visited": nodes_visited,
            }),
        ));
    }
    ranked.sort_by(|left, right| {
        right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1))
    });
    let legal_move_count = ranked.len();
    let moves: Vec<Value> = ranked
        .into_iter()
        .take(top_k)
        .map(|(_, _, payload)| payload)
        .collect();

    Ok(json!({
        "command": "analyze_root_moves",
        "fen": request.fen,
        "requested_nodes_per_move": nodes_per_move,
        "requested_top_k": top_k,
        "legal_move_count": legal_move_count,
        "returned_move_count": moves.len(),
        "total_nodes_visited": total_nodes_visited,
        "moves": moves,
    }))
}

fn response_for(request: &Request, tt: &mut TranspositionTable) -> Result<Value, String> {
    match request.command.as_str() {
        "analyze" => analyze(request, tt),
        "analyze_root_moves" => analyze_root_moves(request, tt),
        "legal_moves" => legal_moves(request),
        "reset" => {
            *tt = TranspositionTable::new();
            Ok(json!({"command": "reset"}))
        }
        "ping" => Ok(json!({"command": "ping", "version": env!("CARGO_PKG_VERSION")})),
        other => Err(format!("unknown command: {other}")),
    }
}

fn emit(mut payload: Value, id: Option<Value>) {
    if let Some(object) = payload.as_object_mut() {
        object.insert("ok".to_owned(), Value::Bool(true));
        if let Some(id) = id {
            object.insert("id".to_owned(), id);
        }
    }
    println!(
        "{}",
        serde_json::to_string(&payload).expect("response should serialize")
    );
    io::stdout().flush().expect("stdout should flush");
}

fn emit_error(message: String, id: Option<Value>) {
    let mut payload = json!({"ok": false, "error": message});
    if let (Some(object), Some(id)) = (payload.as_object_mut(), id) {
        object.insert("id".to_owned(), id);
    }
    println!(
        "{}",
        serde_json::to_string(&payload).expect("error should serialize")
    );
    io::stdout().flush().expect("stdout should flush");
}

fn main() {
    let stdin = io::stdin();
    let mut tt = TranspositionTable::new();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                emit_error(format!("failed to read request: {error}"), None);
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<Request>(&line) {
            Ok(request) => {
                let id = request.id.clone();
                match response_for(&request, &mut tt) {
                    Ok(response) => emit(response, id),
                    Err(error) => emit_error(error, id),
                }
            }
            Err(error) => emit_error(format!("invalid JSON request: {error}"), None),
        }
    }
}
