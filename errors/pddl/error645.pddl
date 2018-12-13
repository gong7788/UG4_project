(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(clear b6)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b0 b7)
		(in-tower b0)
		(on b1 b0)
		(in-tower b1)
		(on b6 b1)
		(in-tower b6)
		(red b0)
		(blue b1)
		(blue b3)
		(red b4)
		(blue b4)
		(red b5)
		(blue b5)
		(red b6)
		(blue b6)
		(red b8)
		(red b9)
		(blue b0)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b6 b7)) (not (on b2 b1)) (not (on b5 b6)) (not (on b4 b6)))))
)