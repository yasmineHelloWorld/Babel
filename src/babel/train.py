from babel.agents.tabular import TabularListener, TabularSpeaker
from babel.config import RunConfig
from babel.datasets.synthetic import COLORS, SHAPES
from babel.env import SyntheticEnvironment
from babel.metrics import evaluate_language
from babel.seeds import make_random_streams
from babel.store import RunStore


def train(config: RunConfig) -> dict:
    config.validate()
    streams = make_random_streams(config.seed)
    environment = SyntheticEnvironment(streams.environment)
    objects = environment.objects
    speaker = TabularSpeaker(len(objects), config.message_length, config.vocab_size, streams.speaker)
    listener = TabularListener(config.message_length, config.vocab_size, len(SHAPES), len(COLORS), streams.listener)
    store = RunStore(config.output_root, config.to_dict(), config.seed)
    baseline = 0.5

    for episode in range(1, config.episodes + 1):
        obj = environment.sample()
        object_id = obj["object_id"]
        message, policy = speaker.speak(object_id)
        shape_probs, color_probs = listener.predict(message)
        reward = 0.5 * (shape_probs[obj["shape_id"]] + color_probs[obj["color_id"]])
        advantage = float(reward - baseline)
        baseline += config.baseline_rate * (float(reward) - baseline)
        speaker.update(object_id, message, policy, advantage, config.learning_rate, config.entropy_bonus)
        listener.observe(message, obj["shape_id"], obj["color_id"])

        if episode % config.log_every == 0 or episode == config.episodes:
            scores = evaluate_language(objects, speaker, listener)
            metric_keys = ("shape_accuracy", "color_accuracy", "exact_accuracy", "message_entropy_bits",
                           "position_entropies_bits", "shape_mutual_information_bits",
                           "color_mutual_information_bits", "position_property_mutual_information_bits",
                           "active_vocabulary", "ambiguity_bits", "synonymy_bits")
            store.log_metrics({"episode": episode, "reward": float(reward), "baseline": baseline,
                               **{key: scores[key] for key in metric_keys}})

    scores = evaluate_language(objects, speaker, listener)
    store.save_checkpoint(speaker_logits=speaker.logits, listener_shape_counts=listener.shape_counts,
                          listener_color_counts=listener.color_counts)
    return {"run_dir": str(store.path), **scores}
