(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b4)
		(clear b4)
		(clear b5)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b3 b7)
		(in-tower b3)
		(on b6 b3)
		(in-tower b6)
		(on b2 b6)
		(in-tower b2)
		(on b5 b2)
		(in-tower b5)
		(red b0)
		(red b4)
		(red b5)
		(blue b0)
		(blue b1)
		(blue b4)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b6 b7)) (not (on b5 b7)) (not (on b4 b7)) (not (on b5 b6)) (not (on b4 b6)) (not (on b4 b5)))))
)