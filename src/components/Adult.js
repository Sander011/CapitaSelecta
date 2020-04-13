import React, { useEffect, useState } from 'react';
import Axios from 'axios';
import {
	Button,
	CircularProgress,
	MenuItem,
	Select,
	Slider,
	Snackbar,
	Typography,
	withStyles,
} from '@material-ui/core';
import Alert from '@material-ui/lab/Alert';

const defaultExplanation = 'Enter feature values and press explain to start!';

const styles = {
	button: {
		width: '10rem',
		height: '3rem',
		padding: '1rem',
		borderRadius: '1rem',
		margin: '1rem',
	},
	container: {
		display: 'flex',
		flexDirection: 'row',
		flexGrow: '1',
		justifyContent: 'space-evenly',
		alignItems: 'center',
		padding: '1rem',
	},
	controls: {
		display: 'flex',
		flexDirection: 'row',
		width: '30vw',
		alignItems: 'center',
		justifyContent: 'space-evenly',
	},
	loadingContainer: {
		display: 'flex',
		flexDirection: 'column',
		alignItems: 'center',
		height: '150px',
		justifyContent: 'space-evenly',
	},
	leftColumn: {
		display: 'flex',
		flexDirection: 'column',
		justifyContent: 'flex-start',
		alignItems: 'flex-start',
		width: '50%',
		height: '70vh',
		margin: '50px',
	},
	rightColumn: {
		display: 'flex',
		flexDirection: 'column',
		justifyContent: 'flex-start',
		alignItems: 'center',
		width: '50%',
		height: '70vh',
		margin: '50px',
	},
	input: {
		display: 'flex',
		width: '100%',
		justifyContent: 'space-between',
		margin: '10px',
	},
	inputText: {
		width: '10vw',
	},
	inputChanger: {
		width: '20vw',
		display: 'flex',
	},
	sliderText: {
		marginLeft: '25px',
		width: '25px',
	},
	options1: {
		display: 'flex',
		flexDirection: 'column',
		alignItems: 'center',
	},
	text: {
		textAlign: 'center',
		margin: '5px',
	},
};

const getFeatures = (id) => Axios.get(`/api/datasets/${id}/retrieve_adult/`);

const Adult = ({ classes }) => {
	const [features, setFeatures] = useState([]);
	const [valuesPerFeature, setValuesPerFeature] = useState({});
	const [boundsPerFeature, setBoundsPerFeature] = useState({});
	const [featureValues, setFeatureValues] = useState({});
	const [error, setError] = useState(undefined);
	const [predicting, setPredicting] = useState(false);
	const [explanation, setExplanation] = useState(defaultExplanation);
	const [userGuess, setUserGuess] = useState(undefined);
	const [prediction, setPrediction] = useState(undefined);
	const [modelUpdated, setModelUpdated] = useState(false);
	const [done, setDone] = useState(false);

	useEffect(() => {
		Axios.all([getFeatures(1590)])
			.then(
				Axios.spread((details) => {
					setFeatures(details.data.features);
					setValuesPerFeature(details.data.values_per_category);
					setBoundsPerFeature(details.data.bounds_per_feature);
				}),
			)
			.catch((_) =>
				setError(
					'Something went wrong while fetching the dataset. Please try another dataset or come back later.',
				),
			);

		return () => {
			setError(undefined);
			setFeatures(undefined);
		};
	}, []);

	const predictSample = () => {
		if (predicting) return;

		setPredicting(true);
		setExplanation(defaultExplanation);
		setPrediction(undefined);
		setUserGuess(undefined);
		setModelUpdated(false);

		const sampleFeatures = {};

		features.forEach((f) => {
			if (f in valuesPerFeature) sampleFeatures[f] = featureValues[f] || valuesPerFeature[f][0];
			if (f in boundsPerFeature) {
				sampleFeatures[f] = featureValues[f] || boundsPerFeature[f][0];
			}
		});

		Axios.get('/api/datasets/1590/predict_adult/', {
			params: {
				sample: sampleFeatures,
			},
		})
			.then((res) => {
				setExplanation(res.data.explanation);
				setPrediction(res.data.prediction);
				setPredicting(false);
			})
			.catch((err) => {
				setError(
					'Something went wrong while predicting the sample. Please try another sample or come back later.',
				);
				setPredicting(false);
			});
	};

	const updateModel = () => {
		const sampleFeatures = {};
		features.forEach((f) => {
			if (f in valuesPerFeature) sampleFeatures[f] = featureValues[f] || valuesPerFeature[f][0];
			if (f in boundsPerFeature) {
				sampleFeatures[f] = featureValues[f] || boundsPerFeature[f][0];
			}
		});

		Axios.get('/api/datasets/1590/update_model/', {
			params: {
				sample: sampleFeatures,
				prediction: userGuess,
			},
		})
			.then((res) => setModelUpdated(true))
			.catch((err) => {
				setError('Something went wrong while updating. Please try again or come back later.');
			});
	};

	const renderValueChangers = (f) => {
		if (f in valuesPerFeature) {
			return (
				<Select
					value={featureValues[f] || valuesPerFeature[f][0]}
					onChange={(e) => handleChange(f, e.target.value)}
					style={styles.inputChanger}
				>
					{valuesPerFeature[f].map((v) => (
						<MenuItem key={v} value={v}>
							{v}
						</MenuItem>
					))}
				</Select>
			);
		} else if (f in boundsPerFeature) {
			const min = parseInt(boundsPerFeature[f][0]);
			const max = parseInt(boundsPerFeature[f][1]);
			return (
				<div style={styles.inputChanger}>
					<Slider
						min={min}
						max={max}
						marks={[
							{ value: min, label: min },
							{ value: max, label: max },
						]}
						step={1.0}
						valueLabelDisplay="auto"
						value={featureValues[f] || min}
						onChange={(_, v) => handleChange(f, v)}
					/>
					<Typography style={styles.sliderText}>{featureValues[f] || min}</Typography>
				</div>
			);
		}
	};

	const handleChange = (f, v) => {
		setPredicting(false);
		setExplanation(defaultExplanation);
		setPrediction(undefined);
		setUserGuess(undefined);

		setFeatureValues({ ...featureValues, [f]: v });
	};

	return (
		<div style={styles.container}>
			<div style={styles.leftColumn}>
				{features.map((f) => (
					<div key={f} style={styles.input}>
						<Typography style={styles.inputText}>{f}</Typography>
						{renderValueChangers(f)}
					</div>
				))}
			</div>
			<div style={styles.rightColumn}>
				<div>
					<Button
						onClick={() => predictSample()}
						className={classes.button}
						variant="contained"
						color="secondary"
					>
						{predicting ? (
							<CircularProgress size={20} color="white" />
						) : (
							<Typography>Explain</Typography>
						)}
					</Button>
				</div>
				<Typography style={styles.text}>
					{predicting
						? 'Explaining sample...'
						: explanation.replace("'1'", 'spam').replace("'0'", 'not spam')}
				</Typography>
				{explanation !== defaultExplanation && (
					<div style={styles.options1}>
						<Typography>What did you think this person was?</Typography>
						<div style={styles.controls}>
							<Button
								className={classes.button}
								variant="contained"
								color={userGuess === '<=50K' ? 'primary' : ''}
								onClick={() => setUserGuess('<=50K')}
							>
								<Typography>{'<=50K'}</Typography>
							</Button>
							<Button
								className={classes.button}
								variant="contained"
								color={userGuess === '>50K' ? 'primary' : ''}
								onClick={() => setUserGuess('>50K')}
							>
								<Typography>{'>50K'}</Typography>
							</Button>
						</div>
						{userGuess &&
							(userGuess !== prediction ? (
								<div style={styles.options1}>
									<Typography>
										You seem to disagree with me, would you like me to use this new information in
										future explanations?
									</Typography>
									<div style={styles.controls}>
										<Button
											className={classes.button}
											variant="contained"
											color="secondary"
											onClick={() => updateModel()}
										>
											<Typography>Yes</Typography>
										</Button>
										<Button
											className={classes.button}
											variant="contained"
											color=""
											onClick={() => setDone(true)}
										>
											<Typography>No</Typography>
										</Button>
									</div>
								</div>
							) : (
								<Typography>
									Great! You seem to agree with me. Enter new details to try again.
								</Typography>
							))}
						<div style={styles.options1}>
							{modelUpdated && (
								<Typography>
									Successfully added your information! Enter new details to try again.
								</Typography>
							)}
							{done && <Typography>That's fine! Enter new details to try again.</Typography>}
						</div>
					</div>
				)}
			</div>
			<Snackbar open={error != null} autoHideDuration={6000} onClose={() => setError(undefined)}>
				<Alert onClose={() => setError(undefined)} severity="error">
					{error}
				</Alert>
			</Snackbar>
		</div>
	);
};

export default withStyles(styles)(Adult);
