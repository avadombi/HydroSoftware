"""TensorFlow implementation of the physics-aware recurrent-neural model family.

The EXP-HYDRO physical recurrent layer follows the differentiable P-RNN
formulation of Jiang, Zheng and Solomatine (2020). HydroSoftware also provides
an HBV recurrent physical layer so that the physical backbone can be selected
without changing the surrounding hybrid architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from hydrosoftware.core.results import SimulationResult
from hydrosoftware.core.optimization import objective_definition, objective_optimization_score
from hydrosoftware.models.neural.tensorflow_models import _tf_objective_loss


def _tf():
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for PaML training and prediction.") from exc
    return tf


def _logit_from_fraction(frac: float) -> float:
    frac = min(max(float(frac), 1e-6), 1.0 - 1e-6)
    return float(np.log(frac / (1.0 - frac)))


def _bounded(raw, lower: float, upper: float):
    tf = _tf()
    return lower + (upper - lower) * tf.math.sigmoid(raw)


class ExpHydroPRNNCell:
    """Factory for a Keras RNN cell reproducing the Jiang et al. P-RNN equations."""

    @staticmethod
    def create():
        tf = _tf()

        class Cell(tf.keras.layers.Layer):
            state_size = (1, 1)
            output_size = 9  # q, snow, soil, snowfall, rain, melt, et, qsub, qsurf

            def build(self, input_shape):
                self.raw_f = self.add_weight(name="raw_f", shape=(), initializer=tf.keras.initializers.Constant(_logit_from_fraction(0.5)), trainable=True)
                self.raw_smax = self.add_weight(name="raw_smax", shape=(), initializer=tf.keras.initializers.Constant(_logit_from_fraction((750.0-100.0)/1400.0)), trainable=True)
                self.raw_qmax = self.add_weight(name="raw_qmax", shape=(), initializer=tf.keras.initializers.Constant(0.0), trainable=True)
                self.raw_ddf = self.add_weight(name="raw_ddf", shape=(), initializer=tf.keras.initializers.Constant(0.0), trainable=True)
                self.raw_tmin = self.add_weight(name="raw_tmin", shape=(), initializer=tf.keras.initializers.Constant(0.0), trainable=True)
                self.raw_tmax = self.add_weight(name="raw_tmax", shape=(), initializer=tf.keras.initializers.Constant(0.0), trainable=True)
                super().build(input_shape)

            @staticmethod
            def heaviside(x):
                return (tf.math.tanh(5.0 * x) + 1.0) / 2.0

            def parameters(self):
                return {
                    "recession_factor": _bounded(self.raw_f, 0.0, 0.1),
                    "maximum_storage": _bounded(self.raw_smax, 100.0, 1500.0),
                    "maximum_subsurface_flow": _bounded(self.raw_qmax, 10.0, 50.0),
                    "degree_day_factor": _bounded(self.raw_ddf, 0.0, 5.0),
                    "snow_temperature_threshold": _bounded(self.raw_tmin, -3.0, 0.0),
                    "melt_temperature_threshold": _bounded(self.raw_tmax, 0.0, 3.0),
                }

            def call(self, inputs, states):
                snow, soil = states
                p = inputs[:, 0:1]
                t = inputs[:, 1:2]
                pet = inputs[:, 2:3]
                par = self.parameters()
                hs = self.heaviside
                snowfall = hs(par["snow_temperature_threshold"] - t) * p
                rain = hs(t - par["snow_temperature_threshold"]) * p
                melt = hs(t - par["melt_temperature_threshold"]) * hs(snow) * tf.minimum(
                    snow,
                    par["degree_day_factor"] * tf.maximum(t - par["melt_temperature_threshold"], 0.0),
                )
                smax = par["maximum_storage"]
                f = par["recession_factor"]
                qmax = par["maximum_subsurface_flow"]
                positive = hs(soil)
                full = hs(soil - smax)
                below = hs(smax - soil)
                et = positive * (full * pet + below * pet * soil / smax)
                qsub = positive * (full * qmax + below * qmax * tf.exp(-f * (smax - soil)))
                qsurf = positive * full * (soil - smax)
                next_snow = snow + snowfall - melt
                next_soil = soil + rain + melt - et - qsub - qsurf
                q = qsub + qsurf
                out = tf.concat([q, next_snow, next_soil, snowfall, rain, melt, et, qsub, qsurf], axis=-1)
                return out, (next_snow, next_soil)

        return Cell()


class HbvPRNNCell:
    """Differentiable daily HBV soil and two-zone response cell."""

    @staticmethod
    def create():
        tf = _tf()

        class Cell(tf.keras.layers.Layer):
            state_size = (1, 1, 1, 1)
            output_size = 16  # q, 4 states, snowfall, rain, melt, soil input, excess, recharge, et, perc, q0, q1, q2

            def build(self, input_shape):
                defaults = {
                    "fc": (50.0, 700.0, 500.0), "beta": (1.0, 6.0, 2.0),
                    "lp": (0.3, 1.0, 0.7), "perc": (0.0, 6.0, 1.5),
                    "uzl": (0.0, 100.0, 30.0), "k0": (0.05, 0.99, 0.30),
                    "k1": (0.01, 0.8, 0.10), "k2": (0.001, 0.15, 0.02),
                    "ddf": (0.0, 5.0, 2.5), "tmin": (-3.0, 0.0, -1.5),
                    "tmax": (0.0, 3.0, 1.5),
                }
                self._spec = defaults
                for name, (lo, hi, default) in defaults.items():
                    frac = (default-lo)/(hi-lo)
                    setattr(self, f"raw_{name}", self.add_weight(name=f"raw_{name}", shape=(), initializer=tf.keras.initializers.Constant(_logit_from_fraction(frac)), trainable=True))
                super().build(input_shape)

            def parameters(self):
                return {name: _bounded(getattr(self, f"raw_{name}"), lo, hi) for name, (lo, hi, _) in self._spec.items()}

            @staticmethod
            def heaviside(x):
                return (tf.math.tanh(5.0 * x) + 1.0) / 2.0

            def call(self, inputs, states):
                snow, soil, upper, lower = states
                p = inputs[:, 0:1]; t = inputs[:, 1:2]; pet = inputs[:, 2:3]
                par = self.parameters(); hs = self.heaviside
                snowfall = hs(par["tmin"] - t) * p
                rain = hs(t - par["tmin"]) * p
                melt = hs(t-par["tmax"]) * hs(snow) * tf.minimum(snow, par["ddf"]*tf.maximum(t-par["tmax"],0.0))
                next_snow = snow + snowfall - melt
                soil_input = rain + melt
                wet = tf.clip_by_value(soil / par["fc"], 0.0, 1.0)
                recharge = soil_input * tf.pow(wet, par["beta"])
                soil_avail = soil + soil_input - recharge
                excess = tf.maximum(soil_avail - par["fc"], 0.0)
                recharge = recharge + excess
                soil_avail = soil_avail - excess
                evap_factor = tf.minimum(tf.maximum(soil / (par["lp"]*par["fc"]), 0.0), 1.0)
                et = tf.minimum(pet * evap_factor, tf.maximum(soil_avail,0.0))
                next_soil = tf.maximum(soil_avail-et,0.0)
                upper_avail = upper + recharge
                perc = tf.minimum(par["perc"], upper_avail)
                u = upper_avail-perc
                q0 = par["k0"]*tf.maximum(u-par["uzl"],0.0)
                q1 = par["k1"]*u
                scale = tf.minimum(1.0, tf.math.divide_no_nan(u, q0+q1))
                q0 *= scale; q1 *= scale
                next_upper = tf.maximum(u-q0-q1,0.0)
                lower_avail = lower+perc
                q2 = tf.minimum(par["k2"]*lower_avail, lower_avail)
                next_lower = tf.maximum(lower_avail-q2,0.0)
                q = q0+q1+q2
                out = tf.concat([q,next_snow,next_soil,next_upper,next_lower,snowfall,rain,melt,soil_input,excess,recharge,et,perc,q0,q1,q2],axis=-1)
                return out,(next_snow,next_soil,next_upper,next_lower)

        return Cell()


class TrainingOnlyScale:
    @staticmethod
    def create(center: np.ndarray, scale: np.ndarray):
        tf = _tf(); center=np.asarray(center,dtype=np.float32); scale=np.asarray(scale,dtype=np.float32)
        class Layer(tf.keras.layers.Layer):
            def call(self, inputs):
                return (inputs-tf.constant(center))/tf.constant(scale)
        return Layer(name="training_scale")


def build_model(physical_model_id: str, center: np.ndarray, scale: np.ndarray, *, filters: int=8, kernel_size: int=10, seed: int=200):
    tf=_tf(); tf.keras.utils.set_random_seed(seed)
    inputs=tf.keras.Input(shape=(None,3),name="meteorological_forcing")
    if physical_model_id=="exp_hydro.mass_conservative.v1":
        cell=ExpHydroPRNNCell.create(); physical=tf.keras.layers.RNN(cell,return_sequences=True,name="exp_hydro_prnn")(inputs)
    elif physical_model_id=="hbv.mass_conservative_daily.v1":
        cell=HbvPRNNCell.create(); physical=tf.keras.layers.RNN(cell,return_sequences=True,name="hbv_prnn")(inputs)
    else:
        raise ValueError(f"Unsupported PaML physical backbone: {physical_model_id}")
    qphys=physical[...,0:1]
    hybrid_inputs=tf.keras.layers.Concatenate(name="physical_neural_concat")([inputs,qphys])
    scaled=TrainingOnlyScale.create(center,scale)(hybrid_inputs)
    x=tf.keras.layers.Conv1D(filters,kernel_size,padding="causal",activation="elu",kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed),name="conv1")(scaled)
    q=tf.keras.layers.Conv1D(1,1,padding="causal",activation="elu",kernel_initializer=tf.keras.initializers.RandomUniform(seed=seed),name="conv2")(x)
    return tf.keras.Model(inputs,[q,physical],name="paml_prnn_cnn"), cell


@dataclass(frozen=True)
class PamlArtifact:
    engine_id: str
    physical_model_id: str
    best_epoch: int
    center: tuple[float,...]
    scale: tuple[float,...]
    weights: tuple[tuple[str, Any],...]
    filters: int
    kernel_size: int
    seed: int

    def as_mapping(self)->dict[str,Any]:
        return {
            "engine_id":self.engine_id,"physical_model_id":self.physical_model_id,"best_epoch":self.best_epoch,
            "center":list(self.center),"scale":list(self.scale),"weights":[{"name":n,"value":v} for n,v in self.weights],
            "filters":self.filters,"kernel_size":self.kernel_size,"seed":self.seed,
        }


@dataclass(frozen=True)
class PamlFitResult:
    artifact:PamlArtifact
    simulation:SimulationResult
    history:tuple[dict[str,float],...]
    partition_masks:Mapping[str,np.ndarray]
    stopped_early:bool
    cancelled:bool


def _nse_tf(obs,pred,mask):
    tf=_tf(); m=tf.cast(mask,tf.bool); o=tf.boolean_mask(obs,m); p=tf.boolean_mask(pred,m)
    den=tf.reduce_sum(tf.square(o-tf.reduce_mean(o)))
    return 1.0-tf.math.divide_no_nan(tf.reduce_sum(tf.square(o-p)),den)


def fit_paml(config, training, precipitation, temperature, pet, observations, *, start_index:int, train_end_index:int, validation_end_index:int, test_end_index:int, warmup_steps:int=0, objective_id:str="nash_sutcliffe_efficiency", cancellation_requested:Callable[[],bool]|None=None, on_epoch:Callable[[int,int,float],None]|None=None):
    """Fit the joint P-RNN + causal-CNN model with chronological partitions.

    Training gradients are evaluated only through the warm-up/training prefix.
    Validation is evaluated through the training+validation prefix so recurrent
    physical states remain continuous.  The test suffix is evaluated once,
    after the validation-selected checkpoint has been restored.
    """
    tf=_tf()
    p=np.asarray(precipitation[start_index:test_end_index],dtype=np.float32)
    t=np.asarray(temperature[start_index:test_end_index],dtype=np.float32)
    e=np.asarray(pet[start_index:test_end_index],dtype=np.float32)
    y=np.asarray(observations[start_index:test_end_index],dtype=np.float32)
    x=np.stack([p,t,e],axis=-1)
    tr=train_end_index-start_index
    va=validation_end_index-start_index
    n=test_end_index-start_index
    train_mask=np.zeros(n,bool); train_mask[warmup_steps:tr]=True
    val_mask=np.zeros(n,bool); val_mask[tr:va]=True
    test_mask=np.zeros(n,bool); test_mask[va:n]=True

    # Jiang-style CNN scaling is frozen from training information only.  The
    # fourth channel is the P-RNN discharge, so its moments are obtained from
    # an untrained physical forward pass over the training prefix only.
    met_center=x[:tr].mean(axis=0)
    met_scale=x[:tr].std(axis=0)
    met_scale=np.where(met_scale>1e-6,met_scale,1.0)
    provisional_center=np.r_[met_center,0.0].astype(np.float32)
    provisional_scale=np.r_[met_scale,1.0].astype(np.float32)
    provisional_model,_=build_model(
        config.physical_model.model_id,
        provisional_center,
        provisional_scale,
        filters=config.cnn_filters,
        kernel_size=config.cnn_kernel_size,
        seed=training.seed,
    )
    train_prefix_tf=tf.convert_to_tensor(x[None,:tr,:])
    _,physical0=provisional_model(train_prefix_tf,training=False)
    qphys0=np.asarray(physical0[0,:,0],dtype=np.float32)
    q0_center=float(np.mean(qphys0))
    q0_scale=float(np.std(qphys0))
    q0_scale=q0_scale if q0_scale>1e-6 else 1.0

    center=np.r_[met_center,q0_center].astype(np.float32)
    scale=np.r_[met_scale,q0_scale].astype(np.float32)
    model,cell=build_model(
        config.physical_model.model_id,
        center,
        scale,
        filters=config.cnn_filters,
        kernel_size=config.cnn_kernel_size,
        seed=training.seed,
    )
    opt=tf.keras.optimizers.Adam(
        learning_rate=training.learning_rate,
        clipnorm=training.gradient_clip_norm,
    )
    if hasattr(opt,"build"):
        opt.build(model.trainable_variables)

    train_x_tf=tf.convert_to_tensor(x[None,:tr,:])
    train_y_tf=tf.convert_to_tensor(y[:tr])
    train_mask_tf=tf.convert_to_tensor(train_mask[:tr])
    validation_x_tf=tf.convert_to_tensor(x[None,:va,:])
    objective=objective_definition(objective_id)

    @tf.function(reduce_retracing=True)
    def train_step():
        with tf.GradientTape() as tape:
            q,_=model(train_x_tf,training=True)
            pred=q[0,:,0]
            train_obs=tf.boolean_mask(train_y_tf,train_mask_tf)
            train_pred=tf.boolean_mask(pred,train_mask_tf)
            loss=_tf_objective_loss(train_obs,train_pred,objective_id)
        grads=tape.gradient(loss,model.trainable_variables)
        pairs=[(g,v) for g,v in zip(grads,model.trainable_variables) if g is not None]
        if not pairs:
            raise RuntimeError("PaML produced no trainable gradients.")
        for gradient,_ in pairs:
            tf.debugging.assert_all_finite(gradient,"PaML produced a non-finite gradient.")
        opt.apply_gradients(pairs)
        return loss

    @tf.function(reduce_retracing=True)
    def evaluate_prefix(values):
        return model(values,training=False)

    best_weights=None
    best_val=(-np.inf if objective.direction=="maximize" else np.inf)
    best_epoch=0
    wait=0
    history=[]
    cancelled=False

    for epoch in range(1,training.epochs+1):
        if cancellation_requested is not None and cancellation_requested():
            cancelled=True
            break
        loss=train_step()
        q_validation,_=evaluate_prefix(validation_x_tf)
        validation_pred_np=np.asarray(q_validation[0,:,0],dtype=float)
        train_score,_=objective_optimization_score(
            y[:tr][train_mask[:tr]],
            validation_pred_np[:tr][train_mask[:tr]],
            objective_id,
        )
        val_score,_=objective_optimization_score(
            y[:va][val_mask[:va]],
            validation_pred_np[val_mask[:va]],
            objective_id,
        )
        history.append({
            "epoch":float(epoch),
            "training_score":float(train_score),
            "validation_score":float(val_score),
            "loss":float(loss.numpy()),
        })
        improved=(val_score>best_val+training.minimum_delta) if objective.direction=="maximize" else (val_score<best_val-training.minimum_delta)
        if improved:
            best_val=val_score
            best_epoch=epoch
            best_weights=model.get_weights()
            wait=0
        else:
            wait+=1
        if on_epoch is not None:
            on_epoch(epoch,training.epochs,best_val)
        if wait>=training.patience:
            break

    if best_weights is not None:
        model.set_weights(best_weights)
    full_x_tf=tf.convert_to_tensor(x[None,...])
    q,phys=evaluate_prefix(full_x_tf)
    q=np.asarray(q[0,:,0],dtype=float)
    phys=np.asarray(phys[0],dtype=float)
    if config.enforce_nonnegative_output:
        q=np.maximum(q,0.0)
    if config.physical_model.model_id=="exp_hydro.mass_conservative.v1":
        state_names=("snow_storage","soil_storage")
        flux_names=(
            "precipitation","rain","snowfall","snowmelt",
            "actual_evapotranspiration","subsurface_flow","surface_flow",
        )
        states=np.vstack([np.zeros((1,2)),phys[:,1:3]])
        fluxes=np.column_stack([
            p, phys[:,4], phys[:,3], phys[:,5], phys[:,6], phys[:,7], phys[:,8]
        ])
    else:
        state_names=("snow_storage","soil_storage","upper_zone_storage","lower_zone_storage")
        flux_names=(
            "precipitation","rain","snowfall","snowmelt","soil_water_input",
            "soil_excess","groundwater_recharge","actual_evapotranspiration",
            "percolation","fast_flow","intermediate_flow","baseflow",
            "unrouted_streamflow",
        )
        states=np.vstack([np.zeros((1,4)),phys[:,1:5]])
        fluxes=np.column_stack([
            p, phys[:,6], phys[:,5], phys[:,7], phys[:,8], phys[:,9],
            phys[:,10], phys[:,11], phys[:,12], phys[:,13], phys[:,14],
            phys[:,15], phys[:,0],
        ])
    physical_q=phys[:,0]
    par={k:float(v.numpy()) for k,v in cell.parameters().items()}
    diagnostics={
        "engine_id":"paml.tensorflow_prnn_cnn.v1",
        "model_id":config.model_id,
        "physical_model_id":config.physical_model.model_id,
        "training_strategy":"joint_end_to_end",
        "physical_parameters":par,
        "cnn_filters":config.cnn_filters,
        "cnn_kernel_size":config.cnn_kernel_size,
        "training_computation":"training_prefix_gradients_validation_prefix_selection",
    }
    sim=SimulationResult(
        outputs=np.column_stack([q,physical_q]),
        states=states,
        fluxes=fluxes,
        output_names=("streamflow","physical_streamflow"),
        state_names=state_names,
        flux_names=flux_names,
        diagnostics=diagnostics,
    )
    weights=tuple((v.name,np.asarray(v.numpy()).tolist()) for v in model.weights)
    artifact=PamlArtifact(
        "paml.tensorflow_prnn_cnn.v1",
        config.physical_model.model_id,
        best_epoch,
        tuple(map(float,center)),
        tuple(map(float,scale)),
        weights,
        config.cnn_filters,
        config.cnn_kernel_size,
        training.seed,
    )
    return PamlFitResult(
        artifact,
        sim,
        tuple(history),
        {"training":train_mask,"validation":val_mask,"test":test_mask},
        len(history)<training.epochs and not cancelled,
        cancelled,
    )


def _artifact_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the stored PaML artifact mapping from a run payload or artifact."""
    if "engine_id" in payload and "weights" in payload:
        return payload
    artifact = payload.get("artifact")
    if isinstance(artifact, Mapping):
        return artifact
    raise ValueError("The PaML artifact is missing from the saved run.")


def simulate_from_artifact(
    payload: Mapping[str, Any],
    precipitation: np.ndarray,
    temperature: np.ndarray,
    pet: np.ndarray,
    *,
    prepend_forcings: Mapping[str, np.ndarray] | None = None,
    enforce_nonnegative_output: bool = True,
) -> SimulationResult:
    """Rebuild a trained PaML model and run it on a causal forcing sequence.

    ``prepend_forcings`` can contain an antecedent sequence. It is evaluated
    together with the forecast sequence and then removed from returned outputs,
    preserving P-RNN state and causal-convolution context without serializing
    implementation-specific recurrent states.
    """
    artifact = _artifact_mapping(payload)
    current = {
        "precipitation": np.asarray(precipitation, dtype=np.float32),
        "temperature": np.asarray(temperature, dtype=np.float32),
        "potential_evapotranspiration": np.asarray(pet, dtype=np.float32),
    }
    if not all(arr.ndim == 1 for arr in current.values()):
        raise ValueError("PaML forcing arrays must be one-dimensional.")
    n = len(current["precipitation"])
    if any(len(arr) != n for arr in current.values()):
        raise ValueError("PaML forcing arrays must have equal length.")
    prefix = 0
    if prepend_forcings is not None:
        prior = {
            name: np.asarray(prepend_forcings[name], dtype=np.float32)
            for name in current
        }
        prefix = len(prior["precipitation"])
        if any(len(arr) != prefix for arr in prior.values()):
            raise ValueError("Antecedent PaML forcing arrays must have equal length.")
        current = {
            name: np.concatenate((prior[name], current[name]))
            for name in current
        }
    x = np.stack(
        [
            current["precipitation"],
            current["temperature"],
            current["potential_evapotranspiration"],
        ],
        axis=-1,
    )
    center = np.asarray(artifact["center"], dtype=np.float32)
    scale = np.asarray(artifact["scale"], dtype=np.float32)
    model, cell = build_model(
        str(artifact["physical_model_id"]),
        center,
        scale,
        filters=int(artifact.get("filters", 8)),
        kernel_size=int(artifact.get("kernel_size", 10)),
        seed=int(artifact.get("seed", 200)),
    )
    # Build variables before restoring ordered weights.
    _ = model(_tf().convert_to_tensor(x[None, ...]), training=False)
    stored = artifact.get("weights")
    if not isinstance(stored, list):
        raise ValueError("The PaML artifact does not contain an ordered weight list.")
    values = [np.asarray(item["value"], dtype=np.float32) for item in stored]
    if len(values) != len(model.get_weights()):
        raise ValueError("The PaML artifact weight count does not match the model.")
    model.set_weights(values)
    q, phys = model(_tf().convert_to_tensor(x[None, ...]), training=False)
    q = np.asarray(q[0, :, 0], dtype=float)[prefix:]
    phys = np.asarray(phys[0], dtype=float)[prefix:]
    if enforce_nonnegative_output:
        q = np.maximum(q, 0.0)
    if str(artifact["physical_model_id"]) == "exp_hydro.mass_conservative.v1":
        state_names = ("snow_storage", "soil_storage")
        flux_names = (
            "precipitation",
            "rain",
            "snowfall",
            "snowmelt",
            "actual_evapotranspiration",
            "subsurface_flow",
            "surface_flow",
        )
        states = np.vstack([np.full((1, 2), np.nan), phys[:, 1:3]])
        fluxes = np.column_stack((
            current["precipitation"][prefix:],
            phys[:, 4],
            phys[:, 3],
            phys[:, 5],
            phys[:, 6],
            phys[:, 7],
            phys[:, 8],
        ))
    else:
        state_names = (
            "snow_storage",
            "soil_storage",
            "upper_zone_storage",
            "lower_zone_storage",
        )
        flux_names = (
            "precipitation",
            "rain",
            "snowfall",
            "snowmelt",
            "soil_water_input",
            "soil_excess",
            "groundwater_recharge",
            "actual_evapotranspiration",
            "percolation",
            "fast_flow",
            "intermediate_flow",
            "baseflow",
            "unrouted_streamflow",
        )
        states = np.vstack([np.full((1, 4), np.nan), phys[:, 1:5]])
        fluxes = np.column_stack((
            current["precipitation"][prefix:],
            phys[:, 6],
            phys[:, 5],
            phys[:, 7],
            phys[:, 8],
            phys[:, 9],
            phys[:, 10],
            phys[:, 11],
            phys[:, 12],
            phys[:, 13],
            phys[:, 14],
            phys[:, 15],
            phys[:, 0],
        ))
    return SimulationResult(
        outputs=np.column_stack((q, phys[:, 0])),
        states=states,
        fluxes=fluxes,
        output_names=("streamflow", "physical_streamflow"),
        state_names=state_names,
        flux_names=flux_names,
        diagnostics={
            "engine_id": "paml.tensorflow_prnn_cnn.v1",
            "physical_model_id": str(artifact["physical_model_id"]),
            "antecedent_rows": prefix,
        },
    )
