(define (problem blocks-problem)
	(:domain blocksworld-unstack)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0 tower0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(clear b4)
		(done t0)
		(in-tower t0 tower0)
		(done tower0)
		(tower tower0)
		(on b9 t0)
		(in-tower b9 tower0)
		(done b9)
		(on b8 b9)
		(in-tower b8 tower0)
		(done b8)
		(on b7 b8)
		(in-tower b7 tower0)
		(done b7)
		(on b6 b7)
		(in-tower b6 tower0)
		(done b6)
		(on b5 b6)
		(in-tower b5 tower0)
		(done b5)
		(on b3 b5)
		(in-tower b3 tower0)
		(done b3)
		(on b2 b3)
		(in-tower b2 tower0)
		(done b2)
		(on b1 b2)
		(in-tower b1 tower0)
		(done b1)
		(on-table b4)
		(clear b1)
		(pink b0)
		(blue b4)
		(= (red-count tower0) 0)
		(= (blue-count tower0) 0)
		(= (green-count tower0) 0)
		(= (yellow-count tower0) 0)
		(= (purple-count tower0) 0)
		(= (orange-count tower0) 0)
		(= (pink-count tower0) 0)
	)
	(:goal (and (forall (?x) (done ?x)) (and (not (on b4 b5)) (not (on b4 b3)) (not (on b4 b2)) (not (on b4 b1)))))
)